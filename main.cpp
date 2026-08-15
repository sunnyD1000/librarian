#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
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

static bool pageWasPresent = false;
static std::chrono::steady_clock::time_point lastCaptureTime{};

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

// Order quad as TL, TR, BR, BL for a stable perspective warp.
static std::vector<cv::Point2f> orderCorners(const std::vector<cv::Point> &pts)
{
	std::vector<cv::Point2f> corner(4);
	std::vector<cv::Point2f> p;
	p.reserve(4);
	for (const auto &pt : pts)
		p.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));

	auto sumCmp = [](const cv::Point2f &a, const cv::Point2f &b) {
		return (a.x + a.y) < (b.x + b.y);
	};
	auto diffCmp = [](const cv::Point2f &a, const cv::Point2f &b) {
		return (a.y - a.x) < (b.y - b.x);
	};

	corner[0] = *std::min_element(p.begin(), p.end(), sumCmp); // TL
	corner[2] = *std::max_element(p.begin(), p.end(), sumCmp); // BR
	corner[1] = *std::min_element(p.begin(), p.end(), diffCmp); // TR
	corner[3] = *std::max_element(p.begin(), p.end(), diffCmp); // BL
	return corner;
}

static cv::Mat extractPaper(const cv::Mat &bgr, const std::vector<cv::Point> &paper)
{
	std::vector<cv::Point2f> src = orderCorners(paper);

	float widthA = cv::norm(src[2] - src[3]);
	float widthB = cv::norm(src[1] - src[0]);
	float heightA = cv::norm(src[1] - src[2]);
	float heightB = cv::norm(src[0] - src[3]);
	int width = std::max(1, static_cast<int>(std::round(std::max(widthA, widthB))));
	int height = std::max(1, static_cast<int>(std::round(std::max(heightA, heightB))));

	std::vector<cv::Point2f> dst = {
		{0.f, 0.f},
		{static_cast<float>(width - 1), 0.f},
		{static_cast<float>(width - 1), static_cast<float>(height - 1)},
		{0.f, static_cast<float>(height - 1)},
	};

	cv::Mat transform = cv::getPerspectiveTransform(src, dst);
	cv::Mat warped;
	cv::warpPerspective(bgr, warped, transform, cv::Size(width, height));
	return warped;
}

static std::string makeCaptureFilename()
{
	using clock = std::chrono::system_clock;
	const auto now = clock::now();
	const std::time_t t = clock::to_time_t(now);
	const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
				now.time_since_epoch()) %
			1000;

	std::tm tm{};
	localtime_r(&t, &tm);

	std::ostringstream name;
	name << "paper_"
	     << std::put_time(&tm, "%Y%m%d_%H%M%S")
	     << '_' << std::setw(3) << std::setfill('0') << ms.count()
	     << ".jpg";
	return name.str();
}

static void savePaperJpeg(const cv::Mat &bgr, const std::vector<cv::Point> &paper)
{
	cv::Mat cropped = extractPaper(bgr, paper);
	if (cropped.empty())
		return;

	const std::string filename = makeCaptureFilename();
	if (cv::imwrite(filename, cropped)) {
		std::cout << "Saved " << filename << " (" << cropped.cols << 'x'
			  << cropped.rows << ')' << std::endl;
	} else {
		std::cerr << "Failed to save " << filename << std::endl;
	}
}

// Find the outer silhouette of a bright sheet (paper).
// Returns true and fills `paper` with 4 corners when found.
static bool findPaper(const cv::Mat &bgr, std::vector<cv::Point> &paper)
{
	cv::Mat gray, mask;
	cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

	cv::threshold(gray, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	double bestArea = 0.0;
	paper.clear();
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

	return !paper.empty() && bestArea >= minArea;
}

static void maybeCapturePage(const cv::Mat &bgr, const std::vector<cv::Point> &paper)
{
	const auto now = std::chrono::steady_clock::now();
	const bool dueForResnap = pageWasPresent &&
				  (now - lastCaptureTime) >= 5s;

	if (!pageWasPresent || dueForResnap) {
		savePaperJpeg(bgr, paper);
		lastCaptureTime = now;
	}

	pageWasPresent = true;
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

		std::vector<cv::Point> paper;
		if (findPaper(bgr, paper)) {
			maybeCapturePage(bgr, paper);

			const std::vector<std::vector<cv::Point>> outline = { paper };
			cv::polylines(bgr, outline, true, cv::Scalar(0, 255, 0), 3);
			for (const auto &pt : paper)
				cv::circle(bgr, pt, 8, cv::Scalar(0, 0, 255), cv::FILLED);
		} else {
			pageWasPresent = false;
		}

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
	// Full-FOV 4:3 binned mode (matches rpicam-still preview framing).
	streamConfig.size = {1296, 972};
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

	ControlList startControls;
	if (auto cropMax = camera->properties().get(properties::ScalerCropMaximum)) {
		Rectangle full = *cropMax;
		startControls.set(controls::ScalerCrop, full);
		std::cout << "ScalerCrop: " << full.toString() << " (max FOV for mode)" << std::endl;
	}

	camera->start(&startControls);
	for (auto &request : requests)
		camera->queueRequest(request.get());

	std::cout << "Streaming — press 'q' in the preview window to quit" << std::endl;
	std::cout << "Page JPEGs are saved in the current directory and kept on exit" << std::endl;

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
