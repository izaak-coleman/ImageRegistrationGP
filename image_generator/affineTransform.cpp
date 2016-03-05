// affineTransform.cpp
// Anthony Flynn
// (12/02/16)
// Input: <ImageToTrasform>
// Takes an input file and generates transformed versions (translation, rotation
// scale, affine warp).

#include <iostream>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

/// Global variables
const char* source_window = "Source image";
const char* translate_window = "Translate";
const char* rotate_window = "Rotate";
const char* scale_window = "Scale";
const char* affine_window = "Affine";


int main( int argc, char** argv ) {
  // Check for correct input parameters:
  if( argc != 2)
    {
      std::cout <<" Usage: affineTransform <ImageToTrasform>" << std::endl;
      return -1;
    }
  
  // Set up warp matrices for various transforms:
  cv::Mat trans_mat( 2, 3, CV_32FC1 ); // pure translation
  cv::Mat rot_mat( 2, 3, CV_32FC1 ); // pure rotation
  cv::Mat scale_mat( 2, 3, CV_32FC1 ); // pure scale
  cv::Mat affine_mat( 2, 3, CV_32FC1 ); // affine warp

  // Set up matrices to hold transformed images
  cv::Mat src, trans_dst, rotate_dst, scale_dst, affine_dst;

  // Load the source image (to be transformed)
  src = cv::imread( argv[1], CV_LOAD_IMAGE_COLOR );

  // Set the dst images to be the same type and size as src
  trans_dst = cv::Mat::zeros( src.rows, src.cols, src.type() );
  rotate_dst = cv::Mat::zeros( src.rows, src.cols, src.type() );
  scale_dst = cv::Mat::zeros( src.rows, src.cols, src.type() );
  affine_dst = cv::Mat::zeros( src.rows, src.cols, src.type() );

  //-------------SECTION TO COMPLETE PURE TRANSLATION-------------------

  int xTfrm = 50; // x coordinates of translation
  int yTfrm = 50; // y coordinates of translation

  trans_mat = (cv::Mat_<double>(2,3) << 1, 0, xTfrm, 0, 1, yTfrm); // define matrix
  
  // Simple translation of src image - saved in trans_dst
  cv::warpAffine( src, trans_dst, trans_mat, trans_dst.size() );

  //-------------SECTION TO COMPLETE PURE ROTATION ---------------------

  // Compute a rotation matrix with respect to the center of the image
  cv::Point center = cv::Point( affine_dst.cols/2, affine_dst.rows/2);//rotation centre
  double angle = -50.0; // Rotation angle of image (degrees)
  double scale = 1.0; // Isotropic scale factor (e.g. 0.6 = 60% shrink)

  // Get the warp matrix with the specifications above
  rot_mat = cv::getRotationMatrix2D( center, angle, scale );

  // Simple rotation of src image - saved in rotate_dst
  cv::warpAffine( src, rotate_dst, rot_mat, rotate_dst.size() );


  //-------------SECTION TO COMPLETE PURE SCALE ---------------------

  // adjust rotation matrix so just shrinks image
  angle = 0;
  scale = 0.6;

  // Get the warp matrix with the revised specifications
  scale_mat = cv::getRotationMatrix2D( center, angle, scale );
   
  // Simple scale
  cv::warpAffine( src, scale_dst, scale_mat, scale_dst.size() );


  //-------------SECTION TO COMPLETE AFFINE TRASFORM -------------------

  // use two triangles to generate warp matrix - can then be applied to image
  cv::Point2f srcTri[3]; //create array of 3 (x,y) coordinates (triangle vertices)
  cv::Point2f dstTri[3]; //create traingle representing mapped position of srcTri

  // Set the 3 points in src image triangle
  srcTri[0] = cv::Point2f( 0,0 ); // top left of image
  srcTri[1] = cv::Point2f( src.cols - 1, 0 ); // top right of image
  srcTri[2] = cv::Point2f( 0, src.rows - 1 ); // bottom left of image

  // Transforms of coordinates hardcoded - can adjust to change warp matrix
  dstTri[0] = cv::Point2f( src.cols*0.0, src.rows*0.1 );
  dstTri[1] = cv::Point2f( src.cols*0.85, src.rows*0.2 );
  dstTri[2] = cv::Point2f( src.cols*0.15, src.rows*0.9 );

  // Get the Affine Transform from three pairs of corresponding points
  affine_mat = cv::getAffineTransform( srcTri, dstTri );

  // Apply the Affine Transform just found to the src image
  cv::warpAffine( src, affine_dst, affine_mat, affine_dst.size() );

  //-------------OUTPUT IMAGES TO SCREEN --------------------------------

  cv::namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  cv::imshow( source_window, src );

  cv::namedWindow( translate_window, CV_WINDOW_AUTOSIZE );
  cv::imshow( translate_window, trans_dst );

  cv::namedWindow( rotate_window, CV_WINDOW_AUTOSIZE );
  cv::imshow( rotate_window, rotate_dst );

  cv::namedWindow( scale_window, CV_WINDOW_AUTOSIZE );
  cv::imshow( scale_window, scale_dst );

  cv::namedWindow( affine_window, CV_WINDOW_AUTOSIZE );
  cv::imshow( affine_window, affine_dst );

  //-------------SAVE IMAGES TO SOURCE FOLDER ---------------------------
  
  std::string filename = argv[1];
  filename = filename.substr(0, (filename.length()-4) ); // remove .extension
  std::string translateFilename = "lowdef_" + filename + "_translate.jpg";
  std::string rotateFilename = "lowdef_" + filename + "_rotate.jpg";
  std::string scaleFilename = "lowdef_" + filename + "_scale.jpg";
  std::string affineFilename = "lowdef_" + filename + "_affine.jpg";

  cv::imwrite( translateFilename.c_str(), trans_dst );
  cv::imwrite( rotateFilename.c_str(), rotate_dst );
  cv::imwrite( scaleFilename.c_str(), scale_dst );
  cv::imwrite( affineFilename.c_str(), affine_dst );

  /// Wait until user exits the program
  cv::waitKey(0);

  return 0;
}

