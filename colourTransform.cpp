// colourTransform.cpp
// Izaak Coleman
// (03/02/16)
// Input <image.x> <X-transform> <Y-transform>
// Takes in a colour image file, splits channels to R and GB and then transforms
// GB and writes results
// Used to quickly generate a pair of images to register.


#include <iostream>
#include <string>
#include <cstdlib>

#include "opencv2/opencv.hpp"


using namespace cv;

int main(int argc, char **argv){


	if (argc != 4) {
		std::cout << "Usage: <exe> <image.jpg> <X-transform> <Y-transform>" << std::endl
							<< "image.jpg, will split and be transformed by X-Y" << std::endl;
	} // Read in image file to be split into BG and R
	Mat src = imread(argv[1], IMREAD_COLOR);

	// Allocate memory for the BG and R versions of src
	Mat red = Mat(src.rows, src.cols, CV_8UC3, Scalar(0,0,0));
	Mat blueGreen = Mat(src.rows, src.cols, CV_8UC3, Scalar(0,0,0));

	for (int row=0; row < src.rows; row++) {
		for (int col=0; col < src.cols; col++) {

			// Extract the pixel vector at row,col
			Vec3b pixel = src.at<Vec3b>(row,col);

			// Store the blue and green values in blueGreen
			blueGreen.at<Vec3b>(row,col)[0] = pixel[0];		// copy blue pixel
			blueGreen.at<Vec3b>(row,col)[1] = pixel[1];		// copy green pixel


			// Store the red values in red
			red.at<Vec3b>(row,col)[2] = pixel[2];					// copy read pixel
		}
	}




	/* Mat to hold BG transform */
	Mat transform = Mat(src.rows, src.cols, CV_8UC3, Scalar(0,0,0));

	// Generate transformation matrix M 
	std::string xString, yString;
	int xTfrm, yTfrm;
	xString = argv[2];   		yString = argv[3];  				// convert to std::string
	xTfrm = atoi(xString.c_str());	yTfrm = atoi(yString.c_str());   // cvt to int

	Mat M = (Mat_<double>(2,3) << 1, 0, xTfrm, 0, 1, yTfrm); // define t.matrix
	warpAffine(blueGreen, transform, M, src.size());				 // transform

	// generate BG and R file names from src file name
	std::string filename = argv[1];
	filename = filename.substr(0, (filename.length()-4) ); // remove .extension
	std::string redStream = filename + "_R.jpg";
	std::string blueGreenStream = filename + "_BG.jpg";

	// write files
	imwrite(redStream.c_str(), red);
	imwrite(blueGreenStream.c_str(), transform);
	
	return 0;
}
