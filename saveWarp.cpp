#include "opencv2/opencv.hpp"
#include <fstream>

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
