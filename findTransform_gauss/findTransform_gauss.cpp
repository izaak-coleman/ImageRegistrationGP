// findTransform_gauss.cpp
// Anthony Flynn
// (04/03/16)
// Input: <TemplateImage> <InputImage> <OutputWarp>
// Takes a template image and input image as alignment, and outputs the
// warp matrix needed to transform the input image to the same coordinates
// as the template image, and also applies the transform to the input image

#include <fstream>
#include "opencv2/opencv.hpp"
 
static int saveWarp(std::string fileName, const cv::Mat& warp, int motionType);

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

  cv::Mat template_image, input_image; //for original template and input images
  cv::Mat tmp, dst_template, dst_input; //for gauss adjusted images

  // Load template image and input image into CV matrices
  template_image = cv::imread( templateImageName, 0 );
  input_image = cv::imread( inputImageName , 0 );

  /*  
  // Save copies of the template_image
  tmp = template_image;
  dst = tmp;

  cv::namedWindow( "Test", CV_WINDOW_AUTOSIZE );
  cv::imshow( "Test", dst );

  /// Loop
  while( true )
  {
    int c;
    c = cv::waitKey(10);

    if( (char)c == 27 )
      { break; }
    if( (char)c == 'u' )
      { cv::pyrUp( tmp, dst, cv::Size( tmp.cols*2, tmp.rows*2 ) );
        printf( "** Zoom In: Image x 2 \n" );
      }
    else if( (char)c == 'd' )
      { cv::pyrDown( tmp, dst, cv::Size( tmp.cols/2, tmp.rows/2 ) );
       printf( "** Zoom Out: Image / 2 \n" );
     }

    cv::imshow( "Test", dst );
    tmp = dst;
  }
  return 0;
  */

  // Create copies of template image and input image, 5 levels down in pyramid
  
  // Save copies of the template_image
  tmp = template_image;
  dst_template = tmp;

  for(int i = 0; i < 3; i++) {
    cv::pyrDown( tmp, dst_template, cv::Size( tmp.cols/2, tmp.rows/2 ) );
    tmp = dst_template;
  }

  tmp = input_image;
  dst_input = tmp;

  for(int i = 0; i < 3; i++) {
    cv::pyrDown( tmp, dst_input, cv::Size( tmp.cols/2, tmp.rows/2 ) );
    tmp = dst_input;
  }

  /*
  cv::namedWindow( "Template", CV_WINDOW_AUTOSIZE );
  cv::imshow( "Template", dst_template );

  cv::namedWindow( "Input", CV_WINDOW_AUTOSIZE );
  cv::imshow( "Input", dst_input );

  cv::waitKey(0);
  */

  // Define motion model
  const int warp_mode = cv::MOTION_AFFINE;
 
  // Set space for warp matrix.
  cv::Mat warp_matrix = cv::Mat::eye(2, 3, CV_32F);
 
  // Set the stopping criteria for the algorithm
  int number_of_iterations = 10000;
  double termination_eps = 1e-10;
  cv::TermCriteria criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
			number_of_iterations, termination_eps);
 

  /*
  // Run find_transformECC to find the warp matrix
  double cc = cv::findTransformECC (
				template_image,
				input_image,
				warp_matrix,
				warp_mode,
				criteria
				);
  */

  double cc = cv::findTransformECC (
				dst_template,
				dst_input,
				warp_matrix,
				warp_mode,
				criteria
				);

  // Reserve a matrix to store the warped image
  cv::Mat warped_image = cv::Mat(template_image.rows, template_image.cols, CV_32FC1);

  // Apply the warp matrix to the input image to produce a warped image 
  // (i.e. aligned to the template image)
  cv::warpAffine(input_image, warped_image, warp_matrix, warped_image.size(), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
 
  // Save values in the warp matrix to the filename provided on command-line
  saveWarp(outputWarpMatrix, warp_matrix, warp_mode);

  std::cout << "Enhanced correlation coefficient between the template image and the final warped input image = " << cc << std::endl; 

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

/* function to save the final values in the warp matrix to fileName */
static int saveWarp(std::string fileName, const cv::Mat& warp, int motionType)
{
  // it saves the raw matrix elements in a file
  CV_Assert(warp.type()==CV_32FC1);

  const float* matPtr = warp.ptr<float>(0);
  int ret_value;

  std::ofstream outfile(fileName.c_str());
  if( !outfile ) {
    std::cerr << "error in saving "
	      << "Couldn't open file '" << fileName.c_str() << "'!" << std::endl;
    ret_value = 0;
  }
  else {//save the warp's elements
    outfile << matPtr[0] << " " << matPtr[1] << " " << matPtr[2] << std::endl;
    outfile << matPtr[3] << " " << matPtr[4] << " " << matPtr[5] << std::endl;

    ret_value = 1;
  }
  return ret_value;

}
