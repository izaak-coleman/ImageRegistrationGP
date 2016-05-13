// GPU Translation of findTransform.cpp
// (29/02/16)
// Original file by Anthony Flynn, annotated by Izaak Coleman and Zoe Vance
// Input: <TemplateImage> <InputImage> <OutputWarp>
// Takes a template image and input image as alignment, and outputs the
// warp matrix needed to transform the input image to the same coordinates
// as the template image, and also applies the transform to the input image

#ifndef FINDTRANSFORMV3FULL_H
#define FINDTRANDFORMV3FULL_H

#include <fstream>
#include "opencv2/opencv.hpp"

using namespace cv;


void saveWarp(std::string fileName, const cv::Mat& warp);
/* saveWarp writes the output warp matrix, to a file. */

// Estimate a transformation matrix for an affine transformation
void image_jacobian_affine_ECC(const cuda::GpuMat& gpu_src1, 
                               const cuda::GpuMat& gpu_src2,
                               const cuda::GpuMat& gpu_src3, 
                               const cuda::GpuMat& gpu_src4,
                               Mat& dst,
                               double &transferTime);


// Calculate the jacobian matrix as step for transformation estimation 
void project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst,
     double &transferTime);

// Update warp matrix with current estimates
void update_warping_matrix_ECC (Mat& map_matrix, const Mat& update);


// compute registration matrix
double gpu_findTransformECC(InputArray templateImage,
         InputArray inputImage,
         InputOutputArray warpMatrix,
         int motionType,
         TermCriteria criteria,
         Mat inputMask,
         double &transferTime
         );

#endif
