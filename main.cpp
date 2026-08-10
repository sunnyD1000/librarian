#include <iostream>
#include <memory>
#include <libcamera/libcamera.h>

using namespace libcamera;

int main() {
		std::unique_ptr<CameraManager> cm = std::make_unique<CameraManager>();
		cm->start();
		
		if (cm->cameras().empty()) {
			std::cerr << "No cameras found!" << std::endl;
			return -1;
		}
		
		std::string cameraId = cm->cameras()[0]->id();
		std::shared_ptr<Camera> camera = cm->get(cameraId);
		camera->acquire();
		
		// Configuration
		std::unique_ptr<CameraConfiguration> config = camera->generateConfiguration({ StreamRole::StillCapture });
		
		StreamConfiguration &streamConfig = config->at(0);
		streamConfig.size.width = 1920;
		streamConfig.size.height = 1080;
		streamConfig.pixelFormat = formats::BGR888;
		
		config->validate();
		camera->configure(config.get());
		
		std::cout << "hello" << std::endl;
		
		// Start
		camera->start();
		std::cout << "Camera running at " << streamConfig.size.toString() << std::endl;
		
		camera->stop();
		camera->release();
		cm->stop();
		return 0;
}
