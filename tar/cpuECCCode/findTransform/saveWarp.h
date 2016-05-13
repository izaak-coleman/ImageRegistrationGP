#ifndef SAVEWARP_H
#define SAVEWARP_H

#include "opencv2/opencv.hpp"

/* Function to save the final values in the warp matrix to fileName */
void saveWarp(std::string fileName, const cv::Mat& warp);

#endif
