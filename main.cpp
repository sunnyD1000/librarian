#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <sys/mman.h>
#include <libcamera/libcamera.h>
#include <opencv2/opencv.hpp>

using namespace libcamera;
using namespace std::chrono_literals;

static std::shared_ptr<Camera> camera;
static std::mutex requestMutex;
static std::queue<Request *> completedRequests;
static std::atomic<bool> running{true};
static unsigned int frameWidth = 0;
static unsigned int frameHeight = 0;
static unsigned int frameStride = 0;
static PixelFormat frameFormat;

// Keep the CameraManager callback light: only hand the request to the
// main thread. Heavy work / OpenCV / queueRequest belong there.
void requestComplete(Request *request)
{
	if (request->status() == Request::RequestCancelled)
		return;

	std::lock_guard<std::mutex> lock(requestMutex);
	completedRequests.push(request);
}

static cv::Mat mapToMat(void *address)
{
	// OpenCV Mat rows/cols are (height, width). Use the real stride so
	// padded rows from the camera do not overrun the buffer.
	if (frameFormat == formats::BGR888) {
		return cv::Mat(frameHeight, frameWidth, CV_8UC3, address, frameStride);
	}
	if (frameFormat == formats::RGB888) {
		return cv::Mat(frameHeight, frameWidth, CV_8UC3, address, frameStride);
	}
	if (frameFormat == formats::XRGB8888 || frameFormat == formats::XBGR8888) {
		return cv::Mat(frameHeight, frameWidth, CV_8UC4, address, frameStride);
	}

	return {};
}

static cv::Mat frameToBgr(const cv::Mat &frame)
{
	cv::Mat bgr;
	if (frameFormat == formats::BGR888) {
		bgr = frame.clone();
	} else if (frameFormat == formats::RGB888) {
		cv::cvtColor(frame, bgr, cv::COLOR_RGB2BGR);
	} else {
		cv::cvtColor(frame, bgr, cv::COLOR_BGRA2BGR);
	}
	return bgr;
}

// Find the largest convex 4-corner contour (paper) and outline it.
static void detectPaperEdges(cv::Mat &bgr)
{
	cv::Mat gray, edges;
	cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
	cv::Canny(gray, edges, 40, 120);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	double bestArea = 0.0;
	std::vector<cv::Point> paper;
	const double minArea = 0.05 * static_cast<double>(frameWidth) * static_cast<double>(frameHeight);

	for (auto &contour : contours) {
		double peri = cv::arcLength(contour, true);
		std::vector<cv::Point> approx;
		cv::approxPolyDP(contour, approx, 0.02 * peri, true);

		if (approx.size() != 4 || !cv::isContourConvex(approx))
			continue;

		double area = std::fabs(cv::contourArea(approx));
		if (area > bestArea) {
			bestArea = area;
			paper = std::move(approx);
		}
	}

	if (!paper.empty() && bestArea >= minArea) {
		const std::vector<std::vector<cv::Point>> outline = { paper };
		cv::polylines(bgr, outline, true, cv::Scalar(0, 255, 0), 3);
		for (const auto &pt : paper)
			cv::circle(bgr, pt, 8, cv::Scalar(0, 0, 255), cv::FILLED);
	}
}

static bool processRequest(Request *request)
{
	const Request::BufferMap &buffers = request->buffers();
	for (auto const &[stream, buffer] : buffers) {
		(void)stream;

		const FrameBuffer::Plane &plane = buffer->planes()[0];
		int fd = plane.fd.get();
		size_t length = plane.length;

		void *address = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		if (address == MAP_FAILED) {
			std::cerr << "Failed to mmap buffer" << std::endl;
			return false;
		}

		cv::Mat frame = mapToMat(address);
		if (frame.empty()) {
			std::cerr << "Unsupported pixel format: " << frameFormat.toString() << std::endl;
			munmap(address, length);
			return false;
		}

		cv::Mat bgr = frameToBgr(frame);
		munmap(address, length);

		detectPaperEdges(bgr);
		cv::imshow("Libcamera OpenCV Stream", bgr);
	}

	request->reuse(Request::ReuseBuffers);
	camera->queueRequest(request);
	return true;
}

int main()
{
	std::unique_ptr<CameraManager> cm = std::make_unique<CameraManager>();
	cm->start();

	if (cm->cameras().empty()) {
		std::cerr << "No cameras found" << std::endl;
		return -1;
	}

	std::string cameraId = cm->cameras()[0]->id();
	camera = cm->get(cameraId);
	camera->acquire();

	std::unique_ptr<CameraConfiguration> config =
		camera->generateConfiguration({ StreamRole::Viewfinder });

	StreamConfiguration &streamConfig = config->at(0);
	streamConfig.size = {1920, 1080};
	// Prefer RGB888; validate() may still adjust size/format on the Pi.
	streamConfig.pixelFormat = formats::RGB888;

	config->validate();
	std::cout << "Validated configuration: " << streamConfig.toString() << std::endl;

	if (camera->configure(config.get()) < 0) {
		std::cerr << "Camera configure failed" << std::endl;
		return -1;
	}

	frameWidth = streamConfig.size.width;
	frameHeight = streamConfig.size.height;
	frameStride = streamConfig.stride;
	frameFormat = streamConfig.pixelFormat;

	Stream *stream = streamConfig.stream();

	FrameBufferAllocator *allocator = new FrameBufferAllocator(camera);
	if (allocator->allocate(stream) < 0) {
		std::cerr << "Can't allocate buffers" << std::endl;
		return -1;
	}

	const std::vector<std::unique_ptr<FrameBuffer>> &buffers = allocator->buffers(stream);
	std::vector<std::unique_ptr<Request>> requests;

	for (const auto &buffer : buffers) {
		std::unique_ptr<Request> request = camera->createRequest();
		if (!request) {
			std::cerr << "Can't create request" << std::endl;
			return -1;
		}

		int ret = request->addBuffer(stream, buffer.get());
		if (ret < 0) {
			std::cerr << "Can't set buffer for request" << std::endl;
			return -1;
		}

		requests.push_back(std::move(request));
	}

	camera->requestCompleted.connect(requestComplete);

	camera->start();
	for (auto &request : requests)
		camera->queueRequest(request.get());

	std::cout << "Streaming — press 'q' in the preview window to quit" << std::endl;

	// Process completed frames on the main thread (safe for OpenCV GUI).
	while (running) {
		Request *request = nullptr;
		{
			std::lock_guard<std::mutex> lock(requestMutex);
			if (!completedRequests.empty()) {
				request = completedRequests.front();
				completedRequests.pop();
			}
		}

		if (request) {
			if (!processRequest(request))
				break;
		} else {
			std::this_thread::sleep_for(1ms);
		}

		int key = cv::waitKey(1);
		if (key == 'q' || key == 'Q')
			running = false;
	}

	camera->requestCompleted.disconnect(requestComplete);
	camera->stop();
	allocator->free(stream);
	delete allocator;
	camera->release();
	camera.reset();
	cm->stop();

	return 0;
}
