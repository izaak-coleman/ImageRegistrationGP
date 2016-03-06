#include "saveWarp.h"
#include "opencv2/opencv.hpp"

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
