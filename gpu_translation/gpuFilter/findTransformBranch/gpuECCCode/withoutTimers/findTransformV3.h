// GPU Translation of findTransform.cpp
// (29/02/16)
// Original file by Anthony Flynn, annotated by Izaak Coleman and Zoe Vance
// Input: <TemplateImage> <InputImage> <OutputWarp>
// Takes a template image and input image as alignment, and outputs the
// warp matrix needed to transform the input image to the same coordinates
// as the template image, and also applies the transform to the input image

#ifndef FINDTRANSFORMV3_H
#define FINDTRANDFORMV3_H

#include <fstream>
#include "opencv2/opencv.hpp"
#include <ctime>


using namespace cv;

void saveWarp(std::string fileName, const cv::Mat& warp);
/* saveWarp writes the output warp matrix, to a file. */

void image_jacobian_affine_ECC(const cuda::GpuMat& gpu_src1, 
															 const cuda::GpuMat& gpu_src2,
                               const cuda::GpuMat& gpu_src3, 
															 const cuda::GpuMat& gpu_src4,
                               Mat& dst);


void project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst);

void update_warping_matrix_ECC (Mat& map_matrix, const Mat& update);

double gpu_findTransformECC(InputArray templateImage,
				 											 InputArray inputImage,
				 											 InputOutputArray warpMatrix,
				 											 int motionType,
				 											 TermCriteria criteria,
				 											 Mat inputMask);

#endif
