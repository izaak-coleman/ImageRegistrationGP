// findTransform_gauss.cpp
// Anthony Flynn
// (31/03/16)
// Input: <TemplateImage> <InputImage> <OutputWarp>
// Takes a template image and input image as alignment, and outputs the
// warp matrix needed to transform the input image to the same coordinates
// as the template image, and also applies the transform to the input image

#include <fstream>
#include <cmath>
#include <ctime>
#include "opencv2/opencv.hpp"
 
void saveWarp(std::string fileName, const cv::Mat& warp);

int main( int argc, char** argv )
{
  // Check correct number of command line arguments
  if( argc != 4)
    {
      std::cout << " Usage: findTransform <TemplateImage> <InputImage> 
                     <OutputWarp.cpp>" << std::endl;
      return -1;
    }

  // Define motion model
  const int warp_mode = cv::MOTION_AFFINE;
 
  // Set space for warp matrix.
  cv::Mat warp_matrix = cv::Mat::eye(2, 3, CV_32F);
 
  // Save file names provided on command line.
  const char* templateImageName = argv[1];
  const char* inputImageName = argv[2];
  const char* outputWarpMatrix = argv[3];

  //Create mat object to hold images:
  cv::Mat template_image, input_image; //for original template and input images
  cv::Mat tmp, gauss_template, gauss_input; //for gauss adjusted images

  // Set the stopping criteria for the algorithm
  int number_of_iterations = 1000;
  double termination_eps = 1e-6;
  cv::TermCriteria criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
			number_of_iterations, termination_eps);
 
  // Load template image and input image into CV matrices
  template_image = cv::imread( templateImageName, 0 );
  input_image = cv::imread( inputImageName , 0 );

  //Create mat object to store the warped image
  cv::Mat warped_image = cv::Mat(template_image.rows, template_image.cols, 
				   CV_32FC1);

  int gauss_level = 4; // start point for number of gauss reductions
  double cc = 0; // correlation coefficient

  std::clock_t begin = clock(); // FOR TESTING PURPOSES

  // Do Gauss image reductions and attempt image aligment.  If ECC is < 0.97
  // re-run test using one fewer Gauss image reductions.  Continue until
  // back to full-sized image (i.e. gauss_level = 0) or CC > 0.97
  do {
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

    // Run find_transformECC to find the warp matrix
    cc = cv::findTransformECC (
			       gauss_template,
			       gauss_input,
			       warp_matrix,
			       warp_mode,
			       criteria
				);

  } while(cc<0.97 && gauss_level > 0);

  std::clock_t end = clock(); // FOR TESTING
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC; // FOR TESTING

  // Adjust warp matrix to take account of reduced image size
  float* matPtr = warp_matrix.ptr<float>(0);
  matPtr[2] = matPtr[2] * pow(2, gauss_level);
  matPtr[5] = matPtr[5] * pow(2, gauss_level);

  //OUTPUT ONCE WARP MATRIX DETERMINED:

  // Apply the warp matrix to the input image to produce a warped image 
  // (i.e. aligned to the template image)
  cv::warpAffine(input_image, warped_image, warp_matrix, 
		warped_image.size(), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
 
  // Save values in the warp matrix to the filename provided on command-line
  saveWarp(outputWarpMatrix, warp_matrix);

  std::cout << "Enhanced correlation coefficient (template / warped input) = "
            << cc << std::endl; 

  std::cout << "Gauss Level = " << gauss_level << std::endl; 

  std::cout << "Time = " << elapsed_secs << std::endl; 

  // Show final output
  cv::namedWindow( "Warped Image", CV_WINDOW_AUTOSIZE );
  cv::namedWindow( "Template Image", CV_WINDOW_AUTOSIZE );
  cv::namedWindow( "Input Image", CV_WINDOW_AUTOSIZE );
  cv::imshow( "Template Image", template_image );
  cv::imshow( "Input Image", input_image );
  cv::imshow( "Warped Image", warped_image);

  // Save warped image
  std::string Filename = "warp_test.jpg";
  cv::imwrite( Filename.c_str(), warped_image );
  cv::waitKey(0);

  return 0;
}

/* function to save the final values in the warp matrix to fileName */
void saveWarp(std::string fileName, const cv::Mat& warp)
{
  // it saves the raw matrix elements in a file
  CV_Assert(warp.type()==CV_32FC1);

  const float* matPtr = warp.ptr<float>(0);

  std::ofstream outfile(fileName.c_str());
  //save the warp's elements
  outfile << matPtr[0] << " " << matPtr[1] << " " << matPtr[2] << std::endl;
  outfile << matPtr[3] << " " << matPtr[4] << " " << matPtr[5] << std::endl;
}
