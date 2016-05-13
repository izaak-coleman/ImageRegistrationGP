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

#include "findTransformV3Full.h"

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
  // Check args 
  if (argc != 4) {
      std::cout << " Usage: findTransform <TemplateImage> " << 
      "<InputImage> <Warp_matrix>" << std::endl;
      return -1;
  }
  
  // Save file names provided on command line.
  const char* templateImageName = argv[1];
  const char* deformedImageName = argv[2];
  const char* outputWarpMatrix = argv[3];

  // Load template image and input image into CV matrices
  cv::Mat template_image, deformed_image;
  template_image = cv::imread( templateImageName, 0 );
  deformed_image  = cv::imread( deformedImageName , 0 );
  
  // Define motion model
  const int warp_mode = cv::MOTION_AFFINE;
 
  // Set space for warp matrix.
  cv::Mat warp_matrix = cv::Mat::eye(2, 3, CV_32F);
 
  // Set the stopping criteria for the algorithm
  int number_of_iterations = 1000;
  double termination_eps = 1e-6;
  cv::TermCriteria criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
      number_of_iterations, termination_eps);

 
  // Run find_transformECC to find the warp matrix
  double totalTime = 0, withoutTransfer = 0, transferTime = 0;
  Mat inputMask;
  std::clock_t eccTimer;
  eccTimer = std::clock();
  double cc = gpu_findTransformECC (
           template_image,
           deformed_image,
           warp_matrix,
           warp_mode,
           criteria,
           inputMask,
           transferTime);
  totalTime = (std::clock() - eccTimer)/(double)CLOCKS_PER_SEC;
  withoutTransfer = totalTime - transferTime;

  std::cout << std::endl << "Total time taken: "  << totalTime << std::endl;
  std::cout << std::endl << "Transfer time taken: "  << transferTime << std::endl;
  std::cout << std::endl << "Time without transfer: "  << withoutTransfer << std::endl;
  std::cout << totalTime << ", " << transferTime << ", " << withoutTransfer << std::endl;

  // Reserve a matrix to store the warped image
  cv::Mat corrected_image = cv::Mat(template_image.rows, template_image.cols, CV_32FC1);

  // Apply the warp matrix to the input image to produce a warped image 
  // (i.e. aligned to the template image)
  cv::warpAffine(deformed_image , corrected_image, 
                warp_matrix, corrected_image.size(), 
                cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
 
  // Save values in the warp matrix to the filename provided on command-line
  saveWarp(outputWarpMatrix, warp_matrix);

  std::cout << "Enhanced correlation coefficient between the template " <<
  "image the corrected image = " << cc << std::endl; 

  // Show final output
  cv::namedWindow( "Corrected Image", CV_WINDOW_AUTOSIZE );
  cv::namedWindow( "Template Image", CV_WINDOW_AUTOSIZE );
  cv::namedWindow( "Deformed Image", CV_WINDOW_AUTOSIZE );
  cv::imshow( "Template Image", template_image );
  cv::imshow( "Deformed Image", deformed_image  );
  cv::imshow( "Corrected Image", corrected_image);
  cv::waitKey(0);

  std::string ofname = "corrected_image.jpg";
  cv::imwrite(ofname, corrected_image);

	std::cout << "Writing corrected image..." << std::endl;
  std::string ofname = "corrected_image.jpg";
  cv::imwrite(ofname, corrected_image);

  return 0;
}
