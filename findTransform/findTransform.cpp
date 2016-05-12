// findTransform.cpp
// Anthony Flynn
// (12/02/16)
// Input: <TemplateImage> <InputImage> <OutputWarp>
// Takes a template image and input image as alignment, and outputs the
// warp matrix needed to transform the input image to the same coordinates
// as the template image, and also applies the transform to the input image

#include <ctime>
#include "opencv2/opencv.hpp"
#include "saveWarp.h"
 
int main( int argc, char** argv )
{
  // Check correct number of command line arguments
  if( argc != 4)
    {
      std::cout << " Usage: findTransform <TemplateImage> <InputImage> "
		<< "<OutputWarp.cpp>" << std::endl;
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
  
  // Define motion model
  const int warp_mode = cv::MOTION_AFFINE;
 
  // Set space for warp matrix.
  cv::Mat warp_matrix = cv::Mat::eye(2, 3, CV_32F);
 
  // Set the stopping criteria for the algorithm
  int number_of_iterations = 1000;
  double termination_eps = 1e-6;
  cv::TermCriteria criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
			number_of_iterations, termination_eps);
 
  std::clock_t begin = clock(); // FOR TESTING PURPOSES

  // Run find_transformECC to find the warp matrix
  double cc = cv::findTransformECC (
				template_image,
				input_image,
				warp_matrix,
				warp_mode,
				criteria
				);

  std::clock_t end = clock(); // FOR TESTING
  double elapsed_secs = (end - begin) / (double) CLOCKS_PER_SEC; // FOR TESTING
 
  // Reserve a matrix to store the warped image
  cv::Mat warped_image = cv::Mat(template_image.rows, template_image.cols, CV_32FC1);

  // Apply the warp matrix to the input image to produce a warped image 
  // (i.e. aligned to the template image)
  cv::warpAffine(input_image, warped_image, warp_matrix, warped_image.size(), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
 
  // Save values in the warp matrix to the filename provided on command-line
  saveWarp(outputWarpMatrix, warp_matrix);

  std::cout << "Enhanced correlation coefficient between the template image "
            << "and the final warped input image = " << cc << std::endl; 

  std::cout << "Time = " << elapsed_secs << std::endl; 

  // Show final output
  cv::namedWindow( "Warped Image", CV_WINDOW_AUTOSIZE );
  cv::namedWindow( "Template Image", CV_WINDOW_AUTOSIZE );
  cv::namedWindow( "Input Image", CV_WINDOW_AUTOSIZE );
  cv::imshow( "Template Image", template_image );
  cv::imshow( "Input Image", input_image );
  cv::imshow( "Warped Image", warped_image); // related to warped image test

  
  std::string Filename = "warp_test.jpg";
  cv::imwrite( Filename.c_str(), warped_image );
  cv::waitKey(0);

  return 0;
}
