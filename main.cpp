#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
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
static Stream *activeStream = nullptr;
static FrameBufferAllocator *allocator = nullptr;
static std::vector<std::unique_ptr<Request>> requests;

static const Size kPreviewSize{1296, 972};
static Size stillSize{2592, 1944};

static bool pageWasPresent = false;
static std::chrono::steady_clock::time_point lastCaptureTime{};
static std::optional<std::vector<cv::Point2f>> pendingHighResCapture;
static std::vector<cv::Mat> keptPhotoFingerprints;

// Mean abs difference on a small normalized gray image; below this => duplicate.
static constexpr double kDuplicateDiffThreshold = 0.06;

void requestComplete(Request *request)
{
	if (request->status() == Request::RequestCancelled)
		return;

	std::lock_guard<std::mutex> lock(requestMutex);
	completedRequests.push(request);
}

static void clearCompletedRequests()
{
	std::lock_guard<std::mutex> lock(requestMutex);
	while (!completedRequests.empty())
		completedRequests.pop();
}

static cv::Mat mapToMat(void *address)
{
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

static bool requestToBgr(Request *request, cv::Mat &bgr)
{
	const Request::BufferMap &buffers = request->buffers();
	if (buffers.empty())
		return false;

	FrameBuffer *buffer = buffers.begin()->second;
	const FrameBuffer::Plane &plane = buffer->planes()[0];
	void *address = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED,
			     plane.fd.get(), 0);
	if (address == MAP_FAILED)
		return false;

	cv::Mat frame = mapToMat(address);
	if (frame.empty()) {
		munmap(address, plane.length);
		return false;
	}

	bgr = frameToBgr(frame);
	munmap(address, plane.length);
	return !bgr.empty();
}

static Request *waitForCompletedRequest(std::chrono::milliseconds timeout)
{
	const auto deadline = std::chrono::steady_clock::now() + timeout;
	while (std::chrono::steady_clock::now() < deadline) {
		{
			std::lock_guard<std::mutex> lock(requestMutex);
			if (!completedRequests.empty()) {
				Request *request = completedRequests.front();
				completedRequests.pop();
				return request;
			}
		}
		std::this_thread::sleep_for(1ms);
	}
	return nullptr;
}

static void stopStream()
{
	if (!camera)
		return;

	camera->stop();
	clearCompletedRequests();

	if (allocator && activeStream) {
		allocator->free(activeStream);
		delete allocator;
	}
	allocator = nullptr;
	activeStream = nullptr;
	requests.clear();
}

static bool startStream(const Size &size, StreamRole role)
{
	std::unique_ptr<CameraConfiguration> config = camera->generateConfiguration({ role });
	StreamConfiguration &streamConfig = config->at(0);
	streamConfig.size = size;
	streamConfig.pixelFormat = formats::RGB888;

	config->validate();
	std::cout << "Configured " << streamConfig.toString() << std::endl;

	if (camera->configure(config.get()) < 0) {
		std::cerr << "Camera configure failed" << std::endl;
		return false;
	}

	frameWidth = streamConfig.size.width;
	frameHeight = streamConfig.size.height;
	frameStride = streamConfig.stride;
	frameFormat = streamConfig.pixelFormat;
	activeStream = streamConfig.stream();

	allocator = new FrameBufferAllocator(camera);
	if (allocator->allocate(activeStream) < 0) {
		std::cerr << "Can't allocate buffers" << std::endl;
		delete allocator;
		allocator = nullptr;
		activeStream = nullptr;
		return false;
	}

	const std::vector<std::unique_ptr<FrameBuffer>> &buffers = allocator->buffers(activeStream);
	for (const auto &buffer : buffers) {
		std::unique_ptr<Request> request = camera->createRequest();
		if (!request || request->addBuffer(activeStream, buffer.get()) < 0) {
			std::cerr << "Can't create request" << std::endl;
			stopStream();
			return false;
		}
		requests.push_back(std::move(request));
	}

	ControlList startControls;
	if (auto cropMax = camera->properties().get(properties::ScalerCropMaximum)) {
		startControls.set(controls::ScalerCrop, *cropMax);
	}

	if (camera->start(&startControls) < 0) {
		std::cerr << "Camera start failed" << std::endl;
		stopStream();
		return false;
	}

	for (auto &request : requests)
		camera->queueRequest(request.get());

	return true;
}

static std::vector<cv::Point2f> normalizeCorners(const std::vector<cv::Point> &paper)
{
	std::vector<cv::Point2f> normalized;
	normalized.reserve(paper.size());
	const float w = static_cast<float>(frameWidth);
	const float h = static_cast<float>(frameHeight);
	for (const auto &pt : paper) {
		normalized.emplace_back(static_cast<float>(pt.x) / w,
					static_cast<float>(pt.y) / h);
	}
	return normalized;
}

static std::vector<cv::Point> denormalizeCorners(const std::vector<cv::Point2f> &normalized)
{
	std::vector<cv::Point> paper;
	paper.reserve(normalized.size());
	for (const auto &pt : normalized) {
		paper.emplace_back(static_cast<int>(std::lround(pt.x * frameWidth)),
				   static_cast<int>(std::lround(pt.y * frameHeight)));
	}
	return paper;
}

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

	corner[0] = *std::min_element(p.begin(), p.end(), sumCmp);
	corner[2] = *std::max_element(p.begin(), p.end(), sumCmp);
	corner[1] = *std::min_element(p.begin(), p.end(), diffCmp);
	corner[3] = *std::max_element(p.begin(), p.end(), diffCmp);
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

static cv::Mat makeFingerprint(const cv::Mat &image)
{
	cv::Mat gray, small, fingerprint;
	if (image.channels() == 1)
		gray = image;
	else
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

	cv::resize(gray, small, cv::Size(64, 64), 0, 0, cv::INTER_AREA);
	small.convertTo(fingerprint, CV_32F, 1.0 / 255.0);
	cv::normalize(fingerprint, fingerprint, 0.0, 1.0, cv::NORM_MINMAX);
	return fingerprint;
}

static bool fingerprintsMatch(const cv::Mat &a, const cv::Mat &b)
{
	if (a.empty() || b.empty() || a.size() != b.size())
		return false;

	const double diff = cv::norm(a, b, cv::NORM_L1) / (a.rows * a.cols);
	return diff < kDuplicateDiffThreshold;
}

static bool matchesPreviousPhoto(const cv::Mat &image)
{
	const cv::Mat fingerprint = makeFingerprint(image);
	for (const auto &previous : keptPhotoFingerprints) {
		if (fingerprintsMatch(fingerprint, previous))
			return true;
	}
	return false;
}

static void rememberPhoto(const cv::Mat &image)
{
	keptPhotoFingerprints.push_back(makeFingerprint(image));
}

static void loadExistingPhotoFingerprints()
{
	namespace fs = std::filesystem;
	std::error_code ec;
	for (const auto &entry : fs::directory_iterator(".", ec)) {
		if (ec || !entry.is_regular_file())
			continue;

		const std::string name = entry.path().filename().string();
		if (name.rfind("paper_", 0) != 0 || entry.path().extension() != ".jpg")
			continue;

		cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
		if (image.empty())
			continue;

		rememberPhoto(image);
		std::cout << "Indexed existing photo " << name << std::endl;
	}
}

static bool savePaperJpegIfUnique(const cv::Mat &cropped)
{
	if (cropped.empty())
		return false;

	const std::string filename = makeCaptureFilename();
	if (!cv::imwrite(filename, cropped)) {
		std::cerr << "Failed to save " << filename << std::endl;
		return false;
	}

	std::cout << "Saved " << filename << " (" << cropped.cols << 'x' << cropped.rows
		  << ')' << std::endl;

	if (matchesPreviousPhoto(cropped)) {
		if (std::remove(filename.c_str()) == 0) {
			std::cout << "Deleted " << filename
				  << " (matches a previous photo)" << std::endl;
		} else {
			std::cerr << "Failed to delete duplicate " << filename << std::endl;
		}
		return false;
	}

	rememberPhoto(cropped);
	return true;
}

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

static void maybeCapturePage(const std::vector<cv::Point> &paper)
{
	if (pendingHighResCapture)
		return;

	const auto now = std::chrono::steady_clock::now();
	const bool dueForResnap = pageWasPresent && (now - lastCaptureTime) >= 5s;

	if (!pageWasPresent || dueForResnap) {
		// Keep paper location in normalized coords across the resolution switch.
		pendingHighResCapture = normalizeCorners(paper);
		lastCaptureTime = now;
	}

	pageWasPresent = true;
}

static bool captureHighResJpeg(const std::vector<cv::Point2f> &normalizedCorners)
{
	std::cout << "Switching to still resolution " << stillSize.toString()
		  << " for capture..." << std::endl;

	stopStream();
	if (!startStream(stillSize, StreamRole::StillCapture)) {
		std::cerr << "Failed to start still stream, restoring preview" << std::endl;
		startStream(kPreviewSize, StreamRole::Viewfinder);
		return false;
	}

	// Skip a couple frames so AE can settle at the new mode.
	cv::Mat bgr;
	bool gotFrame = false;
	for (int i = 0; i < 3; ++i) {
		Request *request = waitForCompletedRequest(2000ms);
		if (!request) {
			std::cerr << "Timed out waiting for still frame" << std::endl;
			break;
		}

		cv::Mat frame;
		if (requestToBgr(request, frame)) {
			bgr = frame;
			gotFrame = true;
		}

		request->reuse(Request::ReuseBuffers);
		camera->queueRequest(request);
	}

	bool saved = false;
	if (gotFrame) {
		std::vector<cv::Point> paper = denormalizeCorners(normalizedCorners);
		cv::Mat cropped = extractPaper(bgr, paper);
		saved = savePaperJpegIfUnique(cropped);
	}

	stopStream();
	if (!startStream(kPreviewSize, StreamRole::Viewfinder)) {
		std::cerr << "Failed to restore preview stream" << std::endl;
		running = false;
		return false;
	}

	std::cout << "Restored preview " << kPreviewSize.toString() << std::endl;
	return saved;
}

static bool processRequest(Request *request)
{
	cv::Mat bgr;
	if (!requestToBgr(request, bgr)) {
		std::cerr << "Failed to map frame" << std::endl;
		request->reuse(Request::ReuseBuffers);
		camera->queueRequest(request);
		return false;
	}

	std::vector<cv::Point> paper;
	if (findPaper(bgr, paper)) {
		maybeCapturePage(paper);

		const std::vector<std::vector<cv::Point>> outline = { paper };
		cv::polylines(bgr, outline, true, cv::Scalar(0, 255, 0), 3);
		for (const auto &pt : paper)
			cv::circle(bgr, pt, 8, cv::Scalar(0, 0, 255), cv::FILLED);
	} else {
		pageWasPresent = false;
	}

	cv::imshow("Libcamera OpenCV Stream", bgr);

	request->reuse(Request::ReuseBuffers);
	camera->queueRequest(request);
	return true;
}

static Size discoverStillSize()
{
	std::unique_ptr<CameraConfiguration> config =
		camera->generateConfiguration({ StreamRole::StillCapture });
	StreamConfiguration &streamConfig = config->at(0);
	streamConfig.pixelFormat = formats::RGB888;
	config->validate();
	return streamConfig.size;
}

int main()
{
	std::unique_ptr<CameraManager> cm = std::make_unique<CameraManager>();
	cm->start();

	if (cm->cameras().empty()) {
		std::cerr << "No cameras found" << std::endl;
		return -1;
	}

	camera = cm->get(cm->cameras()[0]->id());
	camera->acquire();

	stillSize = discoverStillSize();
	std::cout << "Still capture size: " << stillSize.toString() << std::endl;

	loadExistingPhotoFingerprints();

	camera->requestCompleted.connect(requestComplete);

	if (!startStream(kPreviewSize, StreamRole::Viewfinder)) {
		camera->release();
		cm->stop();
		return -1;
	}

	std::cout << "Streaming — press 'q' in the preview window to quit" << std::endl;
	std::cout << "Page JPEGs are saved in the current directory and kept on exit" << std::endl;

	while (running) {
		if (pendingHighResCapture) {
			auto corners = *pendingHighResCapture;
			pendingHighResCapture.reset();
			captureHighResJpeg(corners);
			continue;
		}

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
	stopStream();
	camera->release();
	camera.reset();
	cm->stop();

	return 0;
}
