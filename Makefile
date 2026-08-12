
build: main.cpp
	g++ -std=c++17 main.cpp -o cam_app $(shell pkg-config --cflags --libs libcamera opencv4)
clean: cam_app
	rm cam_app
