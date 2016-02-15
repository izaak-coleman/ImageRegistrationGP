//#include <iostream>
//#include "opencv2/opencv.hpp"
//#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include "opencv2/opencv.hpp"

#include <cstring>

using namespace cv;

int main (int argc, char* argv[])
{
  Mat src;
    // Usage: <cmd> <file_in> <file_out>
    // Read original image

  src = imread(argv[1], IMREAD_UNCHANGED); // read in img

  Mat dest = Mat::zeros(src.rows, src.cols, CV_32FC3); // size of dst
  Mat dx  = (Mat_<double>(3,3) << 0, 2, 0, 2, 5, 2, 0, 2, 0);
  filter2D(src, dest, -1, dx);
  
  namedWindow(argv[1], WINDOW_AUTOSIZE);
  namedWindow("After filter", WINDOW_AUTOSIZE);

  imshow(argv[1], src);
  imshow("After filter", dest);
  waitKey();
  return 0;
}


