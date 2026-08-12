#include <iostream>
#include <memory>
#include <vector>
#include <sys/mman.h>
#include <libcamera/libcamera.h>
#include <opencv2/opencv.hpp>

using namespace libcamera;

void requestComplete(Request *request) {
	if (request->status() == Request::RequestCancelled) {
		return;
	}
	
	// Get buffers attached to the request
	const Request::BufferMap &buffers = request->buffers();
	for (auto const &[stream, buffer] : buffers) {
		
		// Grave DMA file descriptor for the first plane
		const FrameBuffer::Plane &plane = buffer->planes()[0];
		int fd = plane.fd.get();
		size_t length = plane.length;
		
		// Map buffer memory into userspace
		void *address = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		if (address == MAP_FAILED) {
			std::cerr << "Failed to mmap buffer" << std::endl;
			return;
		}
		
		// Wrap mapped memory buffer into an OpenCV Mat
		cv::Mat frame(1080, 1920, CV_8UC3, address);
		
		// Process with OpenCV
		cv::Mat gray;
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		cv::imshow("Libcamera OpenCV Stream", gray);
		cv::waitKey(1);
		
		// Cleanup mapped memory
		munmap(address, length);
	}
	
	// Requeue for continuous streaming
	request->reuse(Request::ReuseBuffers);
}

int main() {
		std::unique_ptr<CameraManager> cm = std::make_unique<CameraManager>();
		cm->start();
		
		if (cm->cameras().empty()) {
			std::cerr << "No cameras found" << std::endl;
			return -1;
		}
		
		std::string cameraId = cm->cameras()[0]->id();
		std::shared_ptr<Camera> camera = cm->get(cameraId);
		camera->acquire();
		
		// Configuration
		std::unique_ptr<CameraConfiguration> config = camera->generateConfiguration({ StreamRole::Viewfinder });
		
		StreamConfiguration &streamConfig = config->at(0);
		streamConfig.size = {1920, 1080};
		streamConfig.pixelFormat = formats::BGR888;
		
		config->validate();
		camera->configure(config.get());
		
		Stream *stream = streamConfig.stream();
		
		// Allocate FrameBuffers
		FrameBufferAllocator *allocator = new FrameBufferAllocator(camera);
		allocator->allocate(stream);
		
		const std::vector<std::unique_ptr<FrameBuffer>> &buffers = allocator->buffers(stream);
		std::vector<std::unique_ptr<Request>> requests;
		
		for (const auto &buffer : buffers) {
			std::unique_ptr<Request> request = camera->createRequest();
			if (!request) {
				std::cerr << "Can't create request" << std::endl;
				return -1;
			}
			
			// Attach buffer for specific stream to this request
			int ret = request->addBuffer(stream, buffer.get());
			if (ret < 0) {
				std::cerr << "Can't set buffer for request" << std::endl;
				return -1;
			}
			
			requests.push_back(std::move(request));
		}
		
		// Connect Request Complete signal (Callback handler)
		camera->requestCompleted.connect([](Request *request) {
			if (request->status() == Request::RequestCancelled) {
				return;
			}
			
			std::cout << "Frame captured successfully" << std::endl;
			
			// Processsing
			
			// Reuse request for the next frame cycle
			request->reuse(Request::ReuseBuffers);
			
			// Re-queue the request to keep the camera streaming
			camera->queueRequest(request);
		});
		
		// Start
		camera->start();
		
		for (auto &request : requests) {
			camera->queueRequest(request.get());
		}
		
		// Camera now streaming and invoking callbacks
		
		camera->stop();
		allocator->free(stream);
		delete allocator;
		camera->release();
		cm->stop();
		
		return 0;
}
