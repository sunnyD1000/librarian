
build: main.cpp
	g++ -std=c++17 main.cpp -o cam_app -I/usr/include/libcamera -lcamera
clean: cam_app
	rm cam_app
