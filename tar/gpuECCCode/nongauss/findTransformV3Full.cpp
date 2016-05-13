// GPU Translation of findTransform.cpp
// (29/02/16)
// Original file by Anthony Flynn, annotated by Izaak Coleman and Zoe Vance
// Input: <TemplateImage> <InputImage> <OutputWarp>
// Takes a template image and input image as alignment, and outputs the
// warp matrix needed to transform the input image to the same coordinates
// as the template image, and also applies the transform to the input image

#include <fstream>
#include "opencv2/opencv.hpp"
#include <ctime>

#include "findTransformV3Full.h"

//cuda include files
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudastereo.hpp"
#include "opencv2/cudawarping.hpp"

using namespace cv;

/* function to save the final values in the warp matrix to fileName */
void saveWarp(std::string fileName, const cv::Mat& warp)
{
  // it saves the raw matrix elements in a file
  CV_Assert(warp.type()==CV_32FC1);

  const float* matPtr = warp.ptr<float>(0);

  std::ofstream outfile(fileName.c_str());
  outfile << matPtr[0] << " " << matPtr[1] << " " << matPtr[2] << std::endl;
  outfile << matPtr[3] << " " << matPtr[4] << " " << matPtr[5] << std::endl;
}


void image_jacobian_affine_ECC(const cuda::GpuMat& gpu_src1, 
                               const cuda::GpuMat& gpu_src2,
                               const cuda::GpuMat& gpu_src3, 
                               const cuda::GpuMat& gpu_src4,
                               Mat& dst, double &transferTime)
{
  
    // ensure parameter validity
    CV_Assert(gpu_src1.size() == gpu_src2.size());
    CV_Assert(gpu_src1.size() == gpu_src3.size());
    CV_Assert(gpu_src1.size() == gpu_src4.size());
    CV_Assert(gpu_src1.rows == dst.rows);
    CV_Assert(dst.cols == (6*gpu_src1.cols));
    CV_Assert(dst.type() == CV_32FC1);

    // upload remaining CPU parameter to GPU
    cuda::GpuMat gpu_dst;
    std::clock_t function;
    function = std::clock();      // CUT TRANFER TIME
    gpu_dst.upload(dst);
    transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;

    const int w = gpu_src1.cols;

    //compute Jacobian blocks (6 blocks)
    // gpu_dst.colRange(0,w) = gpu_src1.mul(gpu_src3);//1
    cuda::multiply(gpu_src1, gpu_src3, gpu_dst.colRange(0,w));//1
    cuda::multiply(gpu_src2, gpu_src3, gpu_dst.colRange(w,2*w));
    cuda::multiply(gpu_src1, gpu_src4, gpu_dst.colRange(2*w, 3*w));
    cuda::multiply(gpu_src2, gpu_src4, gpu_dst.colRange(3*w,4*w));
    gpu_src1.copyTo(gpu_dst.colRange(4*w,5*w));
    gpu_src2.copyTo(gpu_dst.colRange(5*w,6*w));

    function = std::clock();
    gpu_dst.download(dst);
    transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;
}


void project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst, 
                               double &transferTime)
{
    /* this functions is used for two types of projections. If src1.cols ==src.cols
    it does a blockwise multiplication (like in the outer product of vectors)
    of the blocks in matrices src1 and src2 and dst
    has size (number_of_blcks x number_of_blocks), otherwise dst is a vector of size
    (number_of_blocks x 1) since src2 is "multiplied"(dot) with each block of src1.

    The number_of_blocks is equal to the number of parameters we are lloking for
    (i.e. rtanslation:2, euclidean: 3, affine: 6, homography: 8)

    */

    std::clock_t function;
    CV_Assert(src1.rows == src2.rows);
    CV_Assert((src1.cols % src2.cols) == 0);
    int w;

    float* dstPtr = dst.ptr<float>(0);

    function = std::clock();      // CUT TRANFER TIME
    cuda::GpuMat gpu_mat;
    transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;
    double norm;

    if (src1.cols !=src2.cols){//dst.cols==1
        w  = src2.cols;
        for (int i=0; i < dst.rows; i++){
            dstPtr[i] = (float) src2.dot(src1.colRange(i*w,(i+1)*w));
        }
    }

    else {
        CV_Assert(dst.cols == dst.rows); //dst is square (and symmetric)
        w = src2.cols/dst.cols;
        Mat mat;
        for (int i=0; i<dst.rows; i++){
            mat = Mat(src1.colRange(i*w, (i+1)*w));


            function = std::clock();      // CUT TRANFER TIME
            gpu_mat.upload(mat);


            transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;

            norm = cuda::norm(gpu_mat, NORM_L2);

            dstPtr[i*(dst.rows+1)] = (float) pow(norm,2); //diagonal elements

            function = std::clock();      // CUT TRANFER TIME
            gpu_mat.download(mat);
            transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;

            for (int j=i+1; j<dst.cols; j++){ //j starts from i+1
                dstPtr[i*dst.cols+j] = (float) mat.dot(src2.colRange(j*w, (j+1)*w));
                dstPtr[j*dst.cols+i] = dstPtr[i*dst.cols+j]; //due to symmetry
            }
        }
    }
}


void update_warping_matrix_ECC (Mat& map_matrix, const Mat& update)
{
    CV_Assert (map_matrix.type() == CV_32FC1);
    CV_Assert (update.type() == CV_32FC1);
    CV_Assert(map_matrix.rows == 2 && update.rows == 6);
    CV_Assert (update.cols == 1);
    CV_Assert( map_matrix.isContinuous());
    CV_Assert( update.isContinuous() );


    float* mapPtr = map_matrix.ptr<float>(0);
    const float* updatePtr = update.ptr<float>(0);


    mapPtr[0] += updatePtr[0];
    mapPtr[3] += updatePtr[1];
    mapPtr[1] += updatePtr[2];
    mapPtr[4] += updatePtr[3];
    mapPtr[2] += updatePtr[4];
    mapPtr[5] += updatePtr[5];
}


double gpu_findTransformECC(InputArray templateImage,
         InputArray inputImage,
         InputOutputArray warpMatrix,
         int motionType,
         TermCriteria criteria,
         Mat inputMask,
         double &transferTime)
{
    std::clock_t function;
    function = std::clock();      // CUT TRANFER TIME


    Mat src = templateImage.getMat();//template iamge
    Mat dst = inputImage.getMat(); //input image (to be warped)
    Mat map = warpMatrix.getMat(); //warp (transformation)

    CV_Assert(!src.empty());
    CV_Assert(!dst.empty());


    if( ! (src.type()==dst.type()))
        CV_Error( Error::StsUnmatchedFormats, 
        "Both input images must have the same data type" );

    //accept only 1-channel images
    if( src.type() != CV_8UC1 && src.type()!= CV_32FC1)
        CV_Error( Error::StsUnsupportedFormat, 
        "Images must have 8uC1 or 32fC1 type");

    if( map.type() != CV_32FC1)
        CV_Error( Error::StsUnsupportedFormat, 
        "warpMatrix must be single-channel floating-point matrix");

    CV_Assert (map.cols == 3);
    CV_Assert (map.rows == 2 || map.rows ==3);


    CV_Assert (criteria.type & TermCriteria::COUNT || criteria.type & TermCriteria::EPS);
    const int numberOfIterations = (criteria.type & TermCriteria::COUNT) ? 
                                    criteria.maxCount : 200;
    const double termination_eps = (criteria.type & TermCriteria::EPS)   ? 
                                    criteria.epsilon  :  -1;

    int paramTemp = 6;//default: affine

    const int numberOfParameters = paramTemp;

    const int ws = src.cols;
    const int hs = src.rows;
    const int wd = dst.cols;
    const int hd = dst.rows;

    Mat Xcoord = Mat(1, ws, CV_32F);
    Mat Ycoord = Mat(hs, 1, CV_32F);
    Mat Xgrid = Mat(hs, ws, CV_32F);
    Mat Ygrid = Mat(hs, ws, CV_32F);

    float* XcoPtr = Xcoord.ptr<float>(0);
    float* YcoPtr = Ycoord.ptr<float>(0);
    int j;
    for (j=0; j<ws; j++)
        XcoPtr[j] = (float) j;
    for (j=0; j<hs; j++)
        YcoPtr[j] = (float) j;

    repeat(Xcoord, hs, 1, Xgrid);
    repeat(Ycoord, 1, ws, Ygrid);
    cuda::GpuMat gpu_Xgrid, gpu_Ygrid;


    function = std::clock();      // CUT TRANFER TIME
    gpu_Xgrid.upload(Xgrid);
    gpu_Ygrid.upload(Ygrid);
    transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;

    Xcoord.release();
    Ycoord.release();

    Mat templateZM    = Mat(hs, ws, CV_32F);// to store the (smoothed)zero-mean version of template
    Mat templateFloat = Mat(hs, ws, CV_32F);// to store the (smoothed) template
    Mat imageFloat    = Mat(hd, wd, CV_32F);// to store the (smoothed) input image
    Mat imageWarped   = Mat(hs, ws, CV_32F);// to store the warped zero-mean input image
    Mat imageMask    = Mat(hs, ws, CV_8U); //to store the final mask
    
    Mat inputMaskMat = inputMask;
    //to use it for mask warping
    Mat preMask;
    
    if(inputMask.empty())
        preMask = Mat::ones(hd, wd, CV_8U);
    else
        threshold(inputMask, preMask, 0, 1, THRESH_BINARY);

    //gaussian filtering is optional
    src.convertTo(templateFloat, templateFloat.type());
    GaussianBlur(templateFloat, templateFloat, Size(5, 5), 0, 0);

    Mat preMaskFloat;
    preMask.convertTo(preMaskFloat, CV_32F);
    GaussianBlur(preMaskFloat, preMaskFloat, Size(5, 5), 0, 0);
    
    // Change threshold.
    preMaskFloat *= (0.5/0.95);
    
    // Rounding conversion.
    preMaskFloat.convertTo(preMask, preMask.type());
    preMask.convertTo(preMaskFloat, preMaskFloat.type());
    
    dst.convertTo(imageFloat, imageFloat.type());
    GaussianBlur(imageFloat, imageFloat, Size(5, 5), 0, 0);

    // needed matrices for gradients and warped gradients
    Mat gradientX = Mat::zeros(hd, wd, CV_32FC1);
    Mat gradientY = Mat::zeros(hd, wd, CV_32FC1);
    Mat gradientXWarped = Mat(hs, ws, CV_32FC1);
    Mat gradientYWarped = Mat(hs, ws, CV_32FC1);


    // calculate first order image derivatives
    Matx13f dx(-0.5f, 0.0f, 0.5f);

    filter2D(imageFloat, gradientX, -1, dx);
    filter2D(imageFloat, gradientY, -1, dx.t());

    gradientX = gradientX.mul(preMaskFloat);
    gradientY = gradientY.mul(preMaskFloat);

    // matrices needed for solving linear equation system for maximizing ECC
    Mat jacobian                = Mat(hs, ws*numberOfParameters, CV_32F);
    Mat hessian                 = Mat(numberOfParameters, numberOfParameters, CV_32F);
    Mat hessianInv              = Mat(numberOfParameters, numberOfParameters, CV_32F);
    Mat imageProjection         = Mat(numberOfParameters, 1, CV_32F);
    Mat templateProjection      = Mat(numberOfParameters, 1, CV_32F);
    Mat imageProjectionHessian  = Mat(numberOfParameters, 1, CV_32F);
    Mat errorProjection         = Mat(numberOfParameters, 1, CV_32F);

    Mat deltaP = Mat(numberOfParameters, 1, CV_32F);//transformation parameter correction
    Mat error = Mat(hs, ws, CV_32F);//error as 2D matrix

    const int imageFlags = INTER_LINEAR  + WARP_INVERSE_MAP;
    const int maskFlags  = INTER_NEAREST + WARP_INVERSE_MAP;


    // iteratively update map_matrix
    double rho      = -1;
    double last_rho = - termination_eps;

    cuda::GpuMat gpu_imageFloat, gpu_imageWarped;
    cuda::GpuMat gpu_gradientX, gpu_gradientY;
    cuda::GpuMat gpu_gradientXWarped, gpu_gradientYWarped;
    cuda::GpuMat gpu_preMask, gpu_imageMask;

    for (int i = 1; (i <= numberOfIterations) && (fabs(rho-last_rho)>= termination_eps); i++)
    {
      function = std::clock();      // CUT TRANFER TIME
      // upload Mats to gpu 
      gpu_imageFloat.upload(imageFloat);
      gpu_imageMask.upload(imageMask);
      gpu_imageWarped.upload(imageWarped);
      gpu_gradientX.upload(gradientX);
      gpu_gradientY.upload(gradientY);
      gpu_gradientXWarped.upload(gradientXWarped);
      gpu_gradientYWarped.upload(gradientYWarped);
      gpu_preMask.upload(preMask);
      transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;

        // warp-back portion of the inputImage and gradients to the coordinate space of the templateImage
        
      cuda::warpAffine(gpu_imageFloat, gpu_imageWarped, map, 
                       imageWarped.size(), imageFlags);
      cuda::warpAffine(gpu_gradientX,  gpu_gradientXWarped, map, 
                       gradientXWarped.size(), imageFlags);
      cuda::warpAffine(gpu_gradientY,  gpu_gradientYWarped, map, 
                       gradientYWarped.size(), imageFlags);
      cuda::warpAffine(gpu_preMask,    gpu_imageMask, map, 
                       imageMask.size(), maskFlags);
        
      // download modified Mats to cpu versions for subsequent analysis

      function = std::clock();      // CUT TRANFER TIME
      gpu_imageWarped.download(imageWarped);
      gpu_imageMask.download(imageMask);
      transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;
  
      Scalar imgMean, imgStd, tmpMean, tmpStd;
      meanStdDev(imageWarped,   imgMean, imgStd, imageMask);
      meanStdDev(templateFloat, tmpMean, tmpStd, imageMask);

      function = std::clock();      // CUT TRANFER TIME
      gpu_imageWarped.upload(imageWarped);
      gpu_imageMask.upload(imageMask);
      transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;

      // Should scalars be on the GPU? img Mean here 
      cuda::subtract(gpu_imageWarped, imgMean, 
                     gpu_imageWarped, gpu_imageMask);//zero-mean input
      
      cuda::GpuMat gpu_templateZM;
      // Create templateFloat matrix on the GPU
      cuda::GpuMat gpu_templateFloat;

      function = std::clock();      // CUT TRANFER TIME
      gpu_templateFloat.upload(templateFloat);
      transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;
      
      // Initialize templateZM on the CPU
      templateZM = Mat::zeros(templateZM.rows, templateZM.cols, templateZM.type());
      
      function = std::clock();      // CUT TRANFER TIME
      gpu_templateZM.upload(templateZM);
      transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;
      
      cuda::subtract(gpu_templateFloat, tmpMean, 
                     gpu_templateZM,  gpu_imageMask);//zero-mean template
    
      function = std::clock();      // CUT TRANFER TIME
      gpu_gradientX.download(gradientX);
      gpu_gradientY.download(gradientY);
      gpu_imageFloat.download(imageFloat);
      gpu_imageMask.download(imageMask);
      gpu_imageWarped.download(imageWarped);
      gpu_preMask.download(preMask);
      gpu_templateFloat.download(templateFloat);
      gpu_templateZM.download(templateZM);
      transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;

      const double tmpNorm = std::sqrt(cuda::countNonZero(gpu_imageMask) * 
      (tmpStd.val[0])*(tmpStd.val[0]));
      const double imgNorm = std::sqrt(cuda::countNonZero(gpu_imageMask) * 
      (imgStd.val[0])*(imgStd.val[0]));
      
      image_jacobian_affine_ECC(gpu_gradientXWarped, gpu_gradientYWarped, 
      gpu_Xgrid, gpu_Ygrid, jacobian, transferTime);


      function = std::clock();      // CUT TRANFER TIME
      gpu_gradientXWarped.download(gradientXWarped);
      gpu_gradientYWarped.download(gradientYWarped);
      transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;

      // calculate Hessian and its inverse
      project_onto_jacobian_ECC(jacobian, jacobian, hessian, transferTime);

      hessianInv = hessian.inv();

      const double correlation = templateZM.dot(imageWarped);

      // calculate enhanced correlation coefficiont (ECC)->rho
      last_rho = rho;
      rho = correlation/(imgNorm*tmpNorm);
      if (cvIsNaN(rho)) {
        CV_Error(Error::StsNoConv, "NaN encountered.");
      }

      // project images into jacobian
      project_onto_jacobian_ECC( jacobian, imageWarped, 
                                 imageProjection, transferTime);
      project_onto_jacobian_ECC(jacobian, templateZM, 
                                templateProjection, transferTime);


      // calculate the parameter lambda to account for illumination variation
      //imageProjectionHessian = hessianInv*imageProjection;

      // code to initialize gemm
      cuda::GpuMat gpu_hessianInv, gpu_imageProjection, gpu_imageProjectionHessian;

      function = std::clock();      // CUT TRANFER TIME
      gpu_hessianInv.upload(hessianInv);
      gpu_imageProjection.upload(imageProjection);
      gpu_imageProjectionHessian.upload(imageProjectionHessian);
      transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;


      /* -------------- gemm equation --------------------
       *   dst = alpha * src1 * src2 + beta * src3 
       * we only requre src1 * src2 so alpha set to 1
       * src3 is a matrix of 0 and beta set to 1
       */

      // generate src3 
      Mat beta = Mat::zeros(gpu_imageProjectionHessian.rows,
                            gpu_imageProjectionHessian.cols, 
                            gpu_imageProjection.type());
      // upload beta (src3) to gpu
      cuda::GpuMat gpu_beta;

      function = std::clock();      // CUT TRANFER TIME
      gpu_beta.upload(beta);
      transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;

      // mulltiply
      cuda::gemm(gpu_hessianInv, 
                 gpu_imageProjection, 1, 
                 gpu_beta, 1,    // no beta required to array of 0
                 gpu_imageProjectionHessian);

      // download
      function = std::clock();      // CUT TRANFER TIME
      gpu_imageProjectionHessian.download(imageProjectionHessian);
      gpu_imageProjection.download(imageProjection);
      gpu_hessianInv.download(hessianInv);
      transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;

      const double lambda_n = (imgNorm*imgNorm) - imageProjection.dot(imageProjectionHessian);
      const double lambda_d = correlation - templateProjection.dot(imageProjectionHessian);
     
       if (lambda_d <= 0.0)
      {
          rho = -1;
          CV_Error(Error::StsNoConv, "The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped");

      }
     
      const double lambda = (lambda_n/lambda_d);

      //estimate the update step delta_p
      cuda::GpuMat gpu_error;
      
      error = lambda*templateZM;
      
      function = std::clock();      // CUT TRANFER TIME
      gpu_error.upload(error);
      gpu_imageWarped.upload(imageWarped);
      transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;
      
      cuda::subtract(gpu_error, gpu_imageWarped,gpu_error);
      
      function = std::clock();      // CUT TRANFER TIME
      gpu_imageWarped.download(imageWarped);
      gpu_error.download(error);
      transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;
        
      project_onto_jacobian_ECC(jacobian, error, 
                                errorProjection, transferTime);
      
      cuda::GpuMat gpu_errorProjection, gpu_deltaP, gpu_b;

      function = std::clock();      // CUT TRANFER TIME
      gpu_errorProjection.upload(errorProjection);
      gpu_hessianInv.upload(hessianInv);
      gpu_deltaP.upload(deltaP);
      transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;

      // generate betamatrix (src3)
      Mat b = Mat::zeros(deltaP.rows, deltaP.cols, deltaP.type());

      function = std::clock();      // CUT TRANFER TIME
      gpu_b.upload(b);
      transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;

      // multiply matricies
      cuda::gemm(gpu_hessianInv, 
                 gpu_errorProjection,1, 
                 gpu_b, 1, gpu_deltaP);

      // download result
      function = std::clock();      // CUT TRANFER TIME
      gpu_errorProjection.download(errorProjection);
      gpu_hessianInv.download(hessianInv);
      gpu_deltaP.download(deltaP);
      transferTime += (std::clock() - function)/(double)CLOCKS_PER_SEC;


      // update warping matrix
      update_warping_matrix_ECC( map, deltaP);

    }

    // return final correlation coefficient
    return rho;
}

/* End of file. */

