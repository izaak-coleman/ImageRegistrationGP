#include <fstream>
#include <string>
// GPU Translation of findTransform.cpp
// (12/05/16)
// Original file by Anthony Flynn, annotated by Izaak Coleman and Zoe Vance
// Input: <TemplateImage> <InputImage> <OutputWarp>
// Takes a template image and input image as alignment, and outputs the
// warp matrix needed to transform the input image to the same coordinates
// as the template image, and also applies the transform to the input image
#include "opencv2/opencv.hpp"
#include <ctime>

#include "findTransformV3Gauss.h"

//cuda include files
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudastereo.hpp"
#include "opencv2/cudawarping.hpp"


int main( int argc, char** argv )
{
  // Check correct number of command line arguments
  if( argc != 4)
    {
      std::cout << " Usage: findTransform <TemplateImage> <InputImage> <OutputWarp.cpp>" << std::endl;
      return -1;
    }
  
  // Save file names provided on command line.
  const char* templateImageName = argv[1];
  const char* inputImageName = argv[2];
  const char* outputWarpMatrix = argv[3];

  cv::Mat template_image, input_image;

  // Load template image and input image into CV matrices
  template_image = cv::imread( templateImageName, 0 );
  input_image = cv::imread( inputImageName , 0 );
  cv::Mat tmp, gauss_template, gauss_input; //for gauss adjusted images
  
  // Define motion model
  const int warp_mode = cv::MOTION_AFFINE;
 
  // Set space for warp matrix.
  cv::Mat warp_matrix = cv::Mat::eye(2, 3, CV_32F);
 

  // Set the stopping criteria for the algorithm
  int number_of_iterations = 1000;
  double termination_eps = 1e-6;
  cv::TermCriteria criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
			number_of_iterations, termination_eps);

  Mat inputMask;


  int gauss_level = 3; // start point for number of gauss reductions
  double cc = 0; // correlation coefficient

	double totalTime = 0, withoutTransfer = 0, transferTime = 0;
 
  // Run find_transformECC to find the warp matrix
	std::clock_t function;
	function = std::clock();
	do{
		gauss_level--;
    // Save copies of the template_image:
    tmp = template_image;
    gauss_template = tmp;

    for(int i = 0; i < gauss_level; i++) {
      cv::pyrDown( tmp, gauss_template, cv::Size( tmp.cols/2, tmp.rows/2 ) );
      tmp = gauss_template;
    } // each call to pyrDown reduces image size by 1/4

    // Save copies of the input_image:
    tmp = input_image;
    gauss_input = tmp;

    for(int i = 0; i < gauss_level; i++) {
      cv::pyrDown( tmp, gauss_input, cv::Size( tmp.cols/2, tmp.rows/2 ) );
      tmp = gauss_input;
    }

  	cc = gpu_findTransformECC (
					 gauss_template,
					 gauss_input,
					 warp_matrix,
					 warp_mode,
					 criteria,
					 inputMask,
					 transferTime);

	}while(cc<0.97 && gauss_level > 0);
	totalTime = (std::clock() - function)/(double)CLOCKS_PER_SEC;

	withoutTransfer = totalTime - transferTime;

	std::cout << std::endl << "Total time taken: "  << totalTime << std::endl;
	std::cout << std::endl << "Transfer time taken: "  << transferTime << std::endl;
	std::cout << std::endl << "Time without transfer: "  << withoutTransfer << std::endl;
	std::cout << totalTime << ", " << transferTime << ", " << withoutTransfer << std::endl;

	// adjust translation distances from gauss_reductoin
	float* matPtr = warp_matrix.ptr<float>(0);
	matPtr[2] = matPtr[2] * pow(2, gauss_level);
	matPtr[5] = matPtr[5] * pow(2, gauss_level);
  // Reserve a matrix to store the warped image
  cv::Mat warped_image = cv::Mat(template_image.rows, template_image.cols, CV_32FC1);

  // Apply the warp matrix to the input image to produce a warped image 
  // (i.e. aligned to the template image)
  cv::warpAffine(input_image, warped_image, warp_matrix, warped_image.size(), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
 
  // Save values in the warp matrix to the filename provided on command-line
  saveWarp(outputWarpMatrix, warp_matrix);

  std::cout << "Enhanced correlation coefficient between the template image and the final warped input image = " << cc << std::endl; 

  // Show final output
  cv::namedWindow( "Corrected Image", CV_WINDOW_AUTOSIZE );
  cv::namedWindow( "Template Image", CV_WINDOW_AUTOSIZE );
  cv::namedWindow( "Deformed Image", CV_WINDOW_AUTOSIZE );
  cv::imshow( "Template Image", template_image );
  cv::imshow( "Deformed Image", input_image );
  cv::imshow( "Corrected Image", warped_image);
  cv::waitKey(0);

  return 0;
}
