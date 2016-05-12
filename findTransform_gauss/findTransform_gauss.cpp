// findTransform_gauss.cpp
// Anthony Flynn
// (12/05/16)
// Input: <TemplateImage> <DeformedImage> <WarpMatrix>
// Takes a template image and deformed image, and outputs the warp matrix 
// required to transform the deformed image to the same coordinates
// as the template image, and saves the corrected image to corrected_image.jpg

#include <cmath>
#include <ctime>
#include "opencv2/opencv.hpp"
#include "saveWarp.h"
 
int main(int argc, char** argv)
{
  // Check correct number of command line arguments
  if( argc != 4)
    {
      std::cout << "Usage: findTransform_gauss <TemplateImage> <DeformedImage>"
	        << " <WarpMatrix>" << std::endl;
      return -1;
    }

  // Save file names provided on command line.
  const char* templateImageName = argv[1];
  const char* deformedImageName = argv[2];
  const char* outputWarpMatrix = argv[3];

  //Create mat object to hold images:
  cv::Mat template_image, deformed_image; //for template and deformed images
  cv::Mat tmp, gauss_template, gauss_deformed; //for gauss adjusted images

  // Define motion model
  const int warp_mode = cv::MOTION_AFFINE;
 
  // Set space for warp matrix.
  cv::Mat warp_matrix = cv::Mat::eye(2, 3, CV_32F);
 
  // Set the stopping criteria for the algorithm
  int number_of_iterations = 1000;
  double termination_eps = 1e-6;
  cv::TermCriteria criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
			number_of_iterations, termination_eps);
 
  // Load template image and input image into CV matrices
  template_image = cv::imread(templateImageName, 0);
  deformed_image = cv::imread(deformedImageName , 0);

  //Create mat object to store the warped image
  cv::Mat corrected_image = cv::Mat(template_image.rows, template_image.cols, 
				   CV_32FC1);

  int gauss_level = 3; // start point for number of gauss reductions
  double cc = 0; // correlation coefficient

  std::clock_t begin = clock(); // for testing

  // Do Gauss image reductions and attempt image aligment.  If ECC is < 0.97
  // re-run test using one fewer Gauss image reductions.  Continue until
  // back to full-sized image (i.e. gauss_level = 0) or CC > 0.97
  do {
    gauss_level--;

    // Save copies of the template_image:
    tmp = template_image;
    gauss_template = tmp;

    for(int i = 0; i < gauss_level; i++) {
      cv::pyrDown(tmp, gauss_template, cv::Size(tmp.cols/2, tmp.rows/2));
      tmp = gauss_template;
    } // each call to pyrDown reduces image size by 1/4x

    // Save copies of the input_image:
    tmp = deformed_image;
    gauss_deformed = tmp;

    for(int i = 0; i < gauss_level; i++) {
      cv::pyrDown(tmp, gauss_deformed, cv::Size(tmp.cols/2, tmp.rows/2));
      tmp = gauss_deformed;
    }

    // Run find_transformECC to find the warp matrix
    cc = cv::findTransformECC (
			       gauss_template,
			       gauss_deformed,
			       warp_matrix,
			       warp_mode,
			       criteria
				);

  } while(cc<0.97 && gauss_level > 0);

  std::clock_t end = clock(); // for testing
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC; // for testing

  // Adjust warp matrix to take account of reduced image size
  float* matPtr = warp_matrix.ptr<float>(0);
  matPtr[2] = matPtr[2] * pow(2, gauss_level);
  matPtr[5] = matPtr[5] * pow(2, gauss_level);

  //OUTPUT ONCE WARP MATRIX DETERMINED:

  // Apply the warp matrix to the input image to produce a warped image 
  // (i.e. aligned to the template image)
  cv::warpAffine(deformed_image, corrected_image, warp_matrix, 
		corrected_image.size(), cv::INTER_LINEAR + 
		cv::WARP_INVERSE_MAP);
 
  // Save values in the warp matrix to the filename provided on command-line
  saveWarp(outputWarpMatrix, warp_matrix);

  std::cout << "Enhanced correlation coefficient between the template image "
	    << "and the corrected image = " << cc << std::endl; 

  std::cout << "Gauss Level = " << gauss_level << std::endl; 

  std::cout << "Time = " << elapsed_secs << std::endl; 

  // Show final output
  cv::namedWindow( "Template Image", CV_WINDOW_AUTOSIZE );
  cv::namedWindow( "Deformed Image", CV_WINDOW_AUTOSIZE );
  cv::namedWindow( "Corrected Image", CV_WINDOW_AUTOSIZE );
  cv::imshow( "Template Image", template_image );
  cv::imshow( "Deformed Image", deformed_image );
  cv::imshow( "Corrected Image", corrected_image);

  // Save warped image
  std::string Filename = "corrected_image.jpg";
  cv::imwrite( Filename.c_str(), corrected_image );
  cv::waitKey(0);

  return 0;
}

