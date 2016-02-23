// findTransform.cpp
// Anthony Flynn
// (12/02/16)
// Input: <TemplateImage> <InputImage> <OutputWarp>
// Takes a template image and input image as alignment, and outputs the
// warp matrix needed to transform the input image to the same coordinates
// as the template image, and also applies the transform to the input image

#include <fstream>
#include "opencv2/opencv.hpp"

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

static int saveWarp(std::string fileName, const cv::Mat& warp, int motionType);

/**static void image_jacobian_homo_ECC(const Mat& src1, const Mat& src2,
                                    const Mat& src3, const Mat& src4,
					  const Mat& src5, Mat& dst);

static void image_jacobian_euclidean_ECC(const Mat& src1, const Mat& src2,
                                         const Mat& src3, const Mat& src4,
					       const Mat& src5, Mat& dst);
*/
static void image_jacobian_affine_ECC(const Mat& src1, const Mat& src2,
                                      const Mat& src3, const Mat& src4,
                                      Mat& gpu_dst);

//static void image_jacobian_translation_ECC(const Mat& src1, const Mat& src2, Mat& dst);

static void project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst);

static void update_warping_matrix_ECC (Mat& map_matrix, const Mat& update, const int motionType);

double modified_findTransformECC(InputArray templateImage,
				 InputArray inputImage,
				 InputOutputArray warpMatrix,
				 int motionType,
				 TermCriteria criteria,
				 Mat inputMask
				 );

/** static float cuda_dot(const cuda::GpuMat& src1, const cuda::GpuMat& src2); **/

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------

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

  cv::Mat template_image, input_image;

  // Load template image and input image into CV matrices
  template_image = cv::imread( templateImageName, 0 );
  input_image = cv::imread( inputImageName , 0 );
  
  // Define motion model
  const int warp_mode = cv::MOTION_AFFINE;
 
  // Set space for warp matrix.
  cv::Mat warp_matrix = cv::Mat::eye(2, 3, CV_32F);
 
  // Set the stopping criteria for the algorithm
  int number_of_iterations = 3000;
  double termination_eps = 1e-10;
  cv::TermCriteria criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
			number_of_iterations, termination_eps);

  Mat inputMask;
 
  // Run find_transformECC to find the warp matrix
  double cc = modified_findTransformECC (
					 template_image,
					 input_image,
					 warp_matrix,
					 warp_mode,
					 criteria,
					 inputMask);
 
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
  cv::imshow( "Warped Image", warped_image);
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


//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------


static void image_jacobian_homo_ECC(const Mat& src1, const Mat& src2,
                                    const Mat& src3, const Mat& src4,
                                    const Mat& src5, Mat& dst)
{


    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.size() == src3.size());
    CV_Assert(src1.size() == src4.size());

    CV_Assert( src1.rows == dst.rows);
    CV_Assert(dst.cols == (src1.cols*8));
    CV_Assert(dst.type() == CV_32FC1);

    CV_Assert(src5.isContinuous());


    const float* hptr = src5.ptr<float>(0);

    const float h0_ = hptr[0];
    const float h1_ = hptr[3];
    const float h2_ = hptr[6];
    const float h3_ = hptr[1];
    const float h4_ = hptr[4];
    const float h5_ = hptr[7];
    const float h6_ = hptr[2];
    const float h7_ = hptr[5];

    const int w = src1.cols;


    //create denominator for all points as a block
    Mat den_ = src3*h2_ + src4*h5_ + 1.0;//check the time nt i = 0; i < dst.rows; i++){

    //create projected points
    Mat hatX_ = -src3*h0_ - src4*h3_ - h6_;
    divide(hatX_, den_, hatX_);
    Mat hatY_ = -src3*h1_ - src4*h4_ - h7_;
    divide(hatY_, den_, hatY_);


    //instead of dividing each block with den,
    //just pre-devide the block of gradients (it's more efficient)

    Mat src1Divided_;
    Mat src2Divided_;

    divide(src1, den_, src1Divided_);
    divide(src2, den_, src2Divided_);


    //compute Jacobian blocks (8 blocks)

    dst.colRange(0, w) = src1Divided_.mul(src3);//1

    dst.colRange(w,2*w) = src2Divided_.mul(src3);//2

    Mat temp_ = (hatX_.mul(src1Divided_)+hatY_.mul(src2Divided_));
    dst.colRange(2*w,3*w) = temp_.mul(src3);//3

    hatX_.release();
    hatY_.release();

    dst.colRange(3*w, 4*w) = src1Divided_.mul(src4);//4

    dst.colRange(4*w, 5*w) = src2Divided_.mul(src4);//5

    dst.colRange(5*w, 6*w) = temp_.mul(src4);//6

    src1Divided_.copyTo(dst.colRange(6*w, 7*w));//7

    src2Divided_.copyTo(dst.colRange(7*w, 8*w));//8
}

static void image_jacobian_euclidean_ECC(const Mat& src1, const Mat& src2,
                                         const Mat& src3, const Mat& src4,
                                         const Mat& src5, Mat& dst)
{

    CV_Assert( src1.size()==src2.size());
    CV_Assert( src1.size()==src3.size());
    CV_Assert( src1.size()==src4.size());

    CV_Assert( src1.rows == dst.rows);
    CV_Assert(dst.cols == (src1.cols*3));
    CV_Assert(dst.type() == CV_32FC1);

    CV_Assert(src5.isContinuous());

    const float* hptr = src5.ptr<float>(0);

    const float h0 = hptr[0];//cos(theta)
    const float h1 = hptr[3];//sin(theta)

    const int w = src1.cols;

    //create -sin(theta)*X -cos(theta)*Y for all points as a block -> hatX
    Mat hatX = -(src3*h1) - (src4*h0);

    //create cos(theta)*X -sin(theta)*Y for all points as a block -> hatY
    Mat hatY = (src3*h0) - (src4*h1);


    //compute Jacobian blocks (3 blocks)
    dst.colRange(0, w) = (src1.mul(hatX))+(src2.mul(hatY));//1

    src1.copyTo(dst.colRange(w, 2*w));//2
    src2.copyTo(dst.colRange(2*w, 3*w));//3
}


static void image_jacobian_affine_ECC(const Mat& src1, const Mat& src2,
                                      const Mat& src3, const Mat& src4,
																			Mat& dst)
{

    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.size() == src3.size());
    CV_Assert(src1.size() == src4.size());

    CV_Assert(src1.rows == dst.rows);
    CV_Assert(dst.cols == (6*src1.cols));

    CV_Assert(dst.type() == CV_32FC1);

		cuda::GpuMat gpu_src1, gpu_src2, gpu_src3, gpu_src4, gpu_dst;

		gpu_src1.upload(src1);
		gpu_src2.upload(src2);
		gpu_src3.upload(src3);
		gpu_src4.upload(src4);
		gpu_dst.upload(dst);

    const int w = src1.cols;

    //compute Jacobian blocks (6 blocks)
		
    // gpu_dst.colRange(0,w) = gpu_src1.mul(gpu_src3);//1
    cuda::multiply(gpu_src1, gpu_src3, gpu_dst.colRange(0,w));//1
		cuda::multiply(gpu_src2, gpu_src3, gpu_dst.colRange(w,2*w));
		cuda::multiply(gpu_src1, gpu_src4, gpu_dst.colRange(2*w, 3*w));
		cuda::multiply(gpu_src2, gpu_src4, gpu_dst.colRange(3*w,4*w));
		gpu_src1.copyTo(gpu_dst.colRange(4*w,5*w));
		gpu_src2.copyTo(gpu_dst.colRange(5*w,6*w));

		gpu_src1.download(src1);
		gpu_src2.download(src2);
		gpu_src3.download(src3);
		gpu_src4.download(src4);
		gpu_dst.download(dst);
}


static void image_jacobian_translation_ECC(const Mat& src1, const Mat& src2, Mat& dst)
{

    CV_Assert( src1.size()==src2.size());

    CV_Assert( src1.rows == dst.rows);
    CV_Assert(dst.cols == (src1.cols*2));
    CV_Assert(dst.type() == CV_32FC1);

    const int w = src1.cols;

    //compute Jacobian blocks (2 blocks)
    src1.copyTo(dst.colRange(0, w));
    src2.copyTo(dst.colRange(w, 2*w));
}


static void project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst)
{
    /* this functions is used for two types of projections. If src1.cols ==src.cols
    it does a blockwise multiplication (like in the outer product of vectors)
    of the blocks in matrices src1 and src2 and dst
    has size (number_of_blcks x number_of_blocks), otherwise dst is a vector of size
    (number_of_blocks x 1) since src2 is "multiplied"(dot) with each block of src1.

    The number_of_blocks is equal to the number of parameters we are lloking for
    (i.e. rtanslation:2, euclidean: 3, affine: 6, homography: 8)

    */
    CV_Assert(src1.rows == src2.rows);
    CV_Assert((src1.cols % src2.cols) == 0);
    int w;

    float* dstPtr = dst.ptr<float>(0);

		cuda::GpuMat gpu_mat;
		double norm;
		/**
    cuda::GpuMat gpu_src1, gpu_src2, gpu_src3, gpu_dst;
		gpu_src1.upload(src1);
		gpu_src2.upload(src2);
		gpu_dst.upload(dst);
**/    
		if (src1.cols !=src2.cols){//dst.cols==1
        w  = src2.cols;
        for (int i=0; i < dst.rows; i++){
            dstPtr[i] = (float) src2.dot(src1.colRange(i*w,(i+1)*w));
					//	dstPtr[i] = (float) cuda_dot(gpu_src2, gpu_src1.colRange(i*w,(i+1)*w));
				}
    }

    else {
        CV_Assert(dst.cols == dst.rows); //dst is square (and symmetric)
        w = src2.cols/dst.cols;
        Mat mat;
        for (int i=0; i<dst.rows; i++){
            mat = Mat(src1.colRange(i*w, (i+1)*w));
						//dstPtr[i*(dst.rows+1)] = (float) pow(norm(mat),2); //diagonal elements
            gpu_mat.upload(mat);
						norm = cuda::norm(gpu_mat, NORM_L2);
						dstPtr[i*(dst.rows+1)] = (float) pow(norm,2); //diagonal elements
						gpu_mat.download(mat);

            for (int j=i+1; j<dst.cols; j++){ //j starts from i+1
                dstPtr[i*dst.cols+j] = (float) mat.dot(src2.colRange(j*w, (j+1)*w));
                dstPtr[j*dst.cols+i] = dstPtr[i*dst.cols+j]; //due to symmetry
            }
        }
    }
}


static void update_warping_matrix_ECC (Mat& map_matrix, const Mat& update, const int motionType)
{
    CV_Assert (map_matrix.type() == CV_32FC1);
    CV_Assert (update.type() == CV_32FC1);

    CV_Assert (motionType == MOTION_TRANSLATION || motionType == MOTION_EUCLIDEAN ||
        motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY);

    if (motionType == MOTION_HOMOGRAPHY)
        CV_Assert (map_matrix.rows == 3 && update.rows == 8);
    else if (motionType == MOTION_AFFINE)
        CV_Assert(map_matrix.rows == 2 && update.rows == 6);
    else if (motionType == MOTION_EUCLIDEAN)
        CV_Assert (map_matrix.rows == 2 && update.rows == 3);
    else
        CV_Assert (map_matrix.rows == 2 && update.rows == 2);

    CV_Assert (update.cols == 1);

    CV_Assert( map_matrix.isContinuous());
    CV_Assert( update.isContinuous() );


    float* mapPtr = map_matrix.ptr<float>(0);
    const float* updatePtr = update.ptr<float>(0);


    if (motionType == MOTION_TRANSLATION){
        mapPtr[2] += updatePtr[0];
        mapPtr[5] += updatePtr[1];
    }
    if (motionType == MOTION_AFFINE) {
        mapPtr[0] += updatePtr[0];
        mapPtr[3] += updatePtr[1];
        mapPtr[1] += updatePtr[2];
        mapPtr[4] += updatePtr[3];
        mapPtr[2] += updatePtr[4];
        mapPtr[5] += updatePtr[5];
    }
    if (motionType == MOTION_HOMOGRAPHY) {
        mapPtr[0] += updatePtr[0];
        mapPtr[3] += updatePtr[1];
        mapPtr[6] += updatePtr[2];
        mapPtr[1] += updatePtr[3];
        mapPtr[4] += updatePtr[4];
        mapPtr[7] += updatePtr[5];
        mapPtr[2] += updatePtr[6];
        mapPtr[5] += updatePtr[7];
    }
    if (motionType == MOTION_EUCLIDEAN) {
        double new_theta = updatePtr[0];
        new_theta += asin(mapPtr[3]);

        mapPtr[2] += updatePtr[1];
        mapPtr[5] += updatePtr[2];
        mapPtr[0] = mapPtr[4] = (float) cos(new_theta);
        mapPtr[3] = (float) sin(new_theta);
        mapPtr[1] = -mapPtr[3];
    }
}


double modified_findTransformECC(InputArray templateImage,
				 InputArray inputImage,
				 InputOutputArray warpMatrix,
				 int motionType,
				 TermCriteria criteria,
				 Mat inputMask)
{


    Mat src = templateImage.getMat();//template iamge
    Mat dst = inputImage.getMat(); //input image (to be warped)
    Mat map = warpMatrix.getMat(); //warp (transformation)

    CV_Assert(!src.empty());
    CV_Assert(!dst.empty());


    if( ! (src.type()==dst.type()))
        CV_Error( Error::StsUnmatchedFormats, "Both input images must have the same data type" );

    //accept only 1-channel images
    if( src.type() != CV_8UC1 && src.type()!= CV_32FC1)
        CV_Error( Error::StsUnsupportedFormat, "Images must have 8uC1 or 32fC1 type");

    if( map.type() != CV_32FC1)
        CV_Error( Error::StsUnsupportedFormat, "warpMatrix must be single-channel floating-point matrix");

    CV_Assert (map.cols == 3);
    CV_Assert (map.rows == 2 || map.rows ==3);

    CV_Assert (motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY ||
        motionType == MOTION_EUCLIDEAN || motionType == MOTION_TRANSLATION);

    if (motionType == MOTION_HOMOGRAPHY){
        CV_Assert (map.rows ==3);
    }

    CV_Assert (criteria.type & TermCriteria::COUNT || criteria.type & TermCriteria::EPS);
    const int    numberOfIterations = (criteria.type & TermCriteria::COUNT) ? criteria.maxCount : 200;
    const double termination_eps    = (criteria.type & TermCriteria::EPS)   ? criteria.epsilon  :  -1;

    int paramTemp = 6;//default: affine
    switch (motionType){
      case MOTION_TRANSLATION:
          paramTemp = 2;
          break;
      case MOTION_EUCLIDEAN:
          paramTemp = 3;
          break;
      case MOTION_HOMOGRAPHY:
          paramTemp = 8;
          break;
    }


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

		// cudamemcopy
    repeat(Xcoord, hs, 1, Xgrid);
    repeat(Ycoord, 1, ws, Ygrid);

    Xcoord.release();
    Ycoord.release();

    Mat templateZM    = Mat(hs, ws, CV_32F);// to store the (smoothed)zero-mean version of template
    Mat templateFloat = Mat(hs, ws, CV_32F);// to store the (smoothed) template
    Mat imageFloat    = Mat(hd, wd, CV_32F);// to store the (smoothed) input image
    Mat imageWarped   = Mat(hs, ws, CV_32F);// to store the warped zero-mean input image
    Mat imageMask		= Mat(hs, ws, CV_8U); //to store the final mask
    
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
			// upload Mats to gpu 
			gpu_imageFloat.upload(imageFloat);
			gpu_imageMask.upload(imageMask);
			gpu_imageWarped.upload(imageWarped);
			gpu_gradientX.upload(gradientX);
			gpu_gradientY.upload(gradientY);
			gpu_gradientXWarped.upload(gradientXWarped);
			gpu_gradientYWarped.upload(gradientYWarped);
			gpu_preMask.upload(preMask);

        // warp-back portion of the inputImage and gradients to the coordinate space of the templateImage
        
			cuda::warpAffine(gpu_imageFloat, gpu_imageWarped,     map, imageWarped.size(),     imageFlags);
			cuda::warpAffine(gpu_gradientX,  gpu_gradientXWarped, map, gradientXWarped.size(), imageFlags);
			cuda::warpAffine(gpu_gradientY,  gpu_gradientYWarped, map, gradientYWarped.size(), imageFlags);
			cuda::warpAffine(gpu_preMask,    gpu_imageMask,       map, imageMask.size(),       maskFlags);
        
			// download modified Mats to cpu versions for subsequent analysis
			gpu_imageWarped.download(imageWarped);
			gpu_imageMask.download(imageMask);
	
			//TODO cuda::meanStdDev does not take in imageMask parameter, need to understand what that costs us
        Scalar imgMean, imgStd, tmpMean, tmpStd;
				meanStdDev(imageWarped,   imgMean, imgStd, imageMask);  // need to reupload imageMask ?
				meanStdDev(templateFloat, tmpMean, tmpStd, imageMask);

			gpu_imageWarped.upload(imageWarped);
			gpu_imageMask.upload(imageMask);

			// Should scalars be on the GPU? img Mean here 
			cuda::subtract(gpu_imageWarped, imgMean, gpu_imageWarped, gpu_imageMask);//zero-mean input
			
			cuda::GpuMat gpu_templateZM;
			// Create templateFloat matrix on the GPU
			cuda::GpuMat gpu_templateFloat;

			gpu_templateFloat.upload(templateFloat);
			
			// Initialize templateZM on the CPU
			// subtract(imageWarped,   imgMean, imageWarped, imageMask);//zero-mean input
      templateZM = Mat::zeros(templateZM.rows, templateZM.cols, templateZM.type());
      //subtract(templateFloat, tmpMean, templateZM,  imageMask);//zero-mean template
			
			gpu_templateZM.upload(templateZM);
			// gpu_templateZM.setTo(Scalar::all(0));			
			
			cuda::subtract(gpu_templateFloat, tmpMean, gpu_templateZM,  gpu_imageMask);//zero-mean template
		
			gpu_gradientX.download(gradientX);
			gpu_gradientY.download(gradientY);
			gpu_gradientXWarped.download(gradientXWarped);
			gpu_gradientYWarped.download(gradientYWarped);
			gpu_imageFloat.download(imageFloat);
			gpu_imageMask.download(imageMask);
			gpu_imageWarped.download(imageWarped);
			gpu_preMask.download(preMask);
			gpu_templateFloat.download(templateFloat);
			gpu_templateZM.download(templateZM);

			// CUDA sqrt not needed as it does not include  matrix-computation
			// const double tmpNorm = std::sqrt(countNonZero(imageMask)*(tmpStd.val[0])*(tmpStd.val[0]));
			// const double imgNorm = std::sqrt(countNonZero(imageMask)*(imgStd.val[0])*(imgStd.val[0]));


			const double tmpNorm = std::sqrt(cuda::countNonZero(gpu_imageMask)*(tmpStd.val[0])*(tmpStd.val[0]));
			const double imgNorm = std::sqrt(cuda::countNonZero(gpu_imageMask)*(imgStd.val[0])*(imgStd.val[0]));
			
			//cuda::GpuMat gpu_jacobian, gpu_Xgrid, gpu_Ygrid;
			//gpu_jacobian.upload(jacobian);	
			//gpu_Xgrid.upload(Xgrid);
			//gpu_Ygrid.upload(Ygrid);
     
			image_jacobian_affine_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, jacobian);

			// image_jacobian_affine has been switched to CUDA
				// delete or upgrade others

		/**	
        switch (motionType){
            case MOTION_AFFINE:
                image_jacobian_affine_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, jacobian);
                break;
            case MOTION_HOMOGRAPHY:
                image_jacobian_homo_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, map, jacobian);
                break;
            case MOTION_TRANSLATION:
                image_jacobian_translation_ECC(gradientXWarped, gradientYWarped, jacobian);
                break;
            case MOTION_EUCLIDEAN:
                image_jacobian_euclidean_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, map, jacobian);
                break;
				}
				*/

        // calculate Hessian and its inverse
        project_onto_jacobian_ECC(jacobian, jacobian, hessian);

        hessianInv = hessian.inv();

        const double correlation = templateZM.dot(imageWarped);

        // calculate enhanced correlation coefficiont (ECC)->rho
        last_rho = rho;
        rho = correlation/(imgNorm*tmpNorm);
        if (cvIsNaN(rho)) {
          CV_Error(Error::StsNoConv, "NaN encountered.");
        }

        // project images into jacobian
        project_onto_jacobian_ECC( jacobian, imageWarped, imageProjection);
        project_onto_jacobian_ECC(jacobian, templateZM, templateProjection);

				cuda::GpuMat gpu_hessianInv, gpu_imageProjection, gpu_imageProjectionHessian;
				
				//gpu_hessianInv.upload(hessianInv);
				//gpu_imageProjection.upload(imageProjection);
			//	gpu_imageProjectionHessian.upload(imageProjectionHessian);

        // calculate the parameter lambda to account for illumination variation
				imageProjectionHessian = hessianInv*imageProjection;
				//cuda::gemm(gpu_hessianInv, gpu_imageProjection, gpu_imageProjectionHessian)

				//gpu_imageProjectionHessian.download(imageProjectionHessian);
				//gpu_imageProjection.download(imageProjection);
				//gpu_hessianInv.download(hessianInv);

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
				//	gpu_errorInterim;
			
				//cuda::multiply(gpu_templateZM, lambda, gpu_errorInterim);
				
				//error = lambda*templateZM - imageWarped;
				error = lambda*templateZM;
				
				gpu_error.upload(error);
				gpu_imageWarped.upload(imageWarped);
				
				cuda::subtract(gpu_error, gpu_imageWarped,gpu_error);
				
				gpu_imageWarped.download(imageWarped);
				gpu_error.download(error);
					
        project_onto_jacobian_ECC(jacobian, error, errorProjection);
        
				deltaP = hessianInv * errorProjection;
			/**	
				cuda::GpuMat gpu_errorProjection, gpu_deltaP;
				gpu_errorProjection.upload(errorProjection);
				gpu_hessianInv.upload(hessianInv);
				gpu_deltaP.upload(deltaP);		
		
				if(gpu_hessianInv.size() != gpu_errorProjection.size()){
					std::cout << "HERE " ;
				}


				cuda::gemm(gpu_hessianInv, gpu_errorProjection,1, gpu_hessianInv, 0, gpu_deltaP);

				gpu_errorProjection.download(errorProjection);
				gpu_hessianInv.download(hessianInv);
				gpu_deltaP.download(deltaP);
*/
        // update warping matrix
        update_warping_matrix_ECC( map, deltaP, motionType);


    }

    // return final correlation coefficient
    return rho;
}


/* End of file. */
