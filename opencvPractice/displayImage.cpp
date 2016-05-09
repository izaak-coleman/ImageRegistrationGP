#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv){
	if (argc != 2) {
		std::cout << "No file supplied" << std::endl;
		return -1;
	}

	Mat image;
	image = imread(argv[1], 1);

	if (!image.data) {
		std::cout << "File corrupted" << std::endl;
		return -1;
	}

	namedWindow("Display", WINDOW_AUTOSIZE);
	imshow("Display", image);

	waitKey(0);

	cout << "Opencv Version" << CV_VERSION << endl;
	cout << "Maj version " << CV_MAJOR_VERSION << endl;
	cout << "Min version " << CV_MINOR_VERSION << endl;
	return 0;
}
