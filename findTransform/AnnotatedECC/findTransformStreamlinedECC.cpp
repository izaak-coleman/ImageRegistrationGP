// findTransform.cpp with annottated eccCPUAlgo
// Anthony Flynn adjusted by Zoe Vance based on Open CV doc
// (15/02/16)
// Input: <TemplateImage> <InputImage> <OutputWarp>
// Takes a template image and input image as alignment, and outputs the
// warp matrix needed to transform the input image to the same coordinates
// as the template image, and also applies the transform to the input image

#include <fstream>
#include "opencv2/opencv.hpp"
using namespace cv;
 
// This version of findStreamlinedECC does NOT take in a sixth mask parameter
double findTransformStreamlinedECC(InputArray templateImage,
                            InputArray inputImage,
                            InputOutputArray warpMatrix,
                            int motionType,
                            TermCriteria criteria);

static void project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst);

static void image_jacobian_affine_ECC(const Mat& src1, const Mat& src2,
                                      const Mat& src3, const Mat& src4,
                                      Mat& dst);

static void update_warping_matrix_ECC (Mat& map_matrix, const Mat& update, const int motionType);

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
 
  // Run find_transformECC to find the warp matrix
  double cc = findTransformStreamlinedECC (
				template_image,
				input_image,
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

		/** Non-Affine 
    if (motionType == MOTION_TRANSLATION){
        mapPtr[2] += updatePtr[0];
        mapPtr[5] += updatePtr[1];
    }
		**/
    if (motionType == MOTION_AFFINE) {
        mapPtr[0] += updatePtr[0];
        mapPtr[3] += updatePtr[1];
        mapPtr[1] += updatePtr[2];
        mapPtr[4] += updatePtr[3];
        mapPtr[2] += updatePtr[4];
        mapPtr[5] += updatePtr[5];
    }
		/** Non-affine
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
		**/
}

// Calculate affine Jacboian transformation 


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


    const int w = src1.cols;

    //compute Jacobian blocks (6 blocks)

    dst.colRange(0,w) = src1.mul(src3);//1
    dst.colRange(w,2*w) = src2.mul(src3);//2
    dst.colRange(2*w,3*w) = src1.mul(src4);//3
    dst.colRange(3*w,4*w) = src2.mul(src4);//4
    src1.copyTo(dst.colRange(4*w,5*w));//5
    src2.copyTo(dst.colRange(5*w,6*w));//6
}


static void project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst)
{
    /* this functions is used for two types of projections. If src1.cols ==src.cols
    it does a blockwise multiplication (like in the outer product of vectors)
    of the blocks in matrices src1 and src2 and dst
    has size (number_of_blcks x number_of_blocks), otherwise dst is a vector of size
    (number_of_blocks x 1) since src2 is "multiplied"(dot) with each block of src1.

    The number_of_blocks is equal to the number of parameters we are lloking for
    (i.e. translation:2, euclidean: 3, affine: 6, homography: 8)

    */

		// Check inputs
    CV_Assert(src1.rows == src2.rows);
    CV_Assert((src1.cols % src2.cols) == 0);
    
		int w;

    float* dstPtr = dst.ptr<float>(0);

    if (src1.cols !=src2.cols){//dst.cols==1
        w  = src2.cols;
        for (int i=0; i<dst.rows; i++){
            dstPtr[i] = (float) src2.dot(src1.colRange(i*w,(i+1)*w));
        }
    }

    else {
        CV_Assert(dst.cols == dst.rows); //dst is square (and symmetric)
        w = src2.cols/dst.cols;
        Mat mat;
        for (int i=0; i<dst.rows; i++){

            mat = Mat(src1.colRange(i*w, (i+1)*w));
            dstPtr[i*(dst.rows+1)] = (float) pow(norm(mat),2); //diagonal elements

            for (int j=i+1; j<dst.cols; j++){ //j starts from i+1
                dstPtr[i*dst.cols+j] = (float) mat.dot(src2.colRange(j*w, (j+1)*w));
                dstPtr[j*dst.cols+i] = dstPtr[i*dst.cols+j]; //due to symmetry
            }
        }
    }
}



/**
// main function
Inputs:
InputArray is a proxy class for read-only input arrays. Constructed of/can pass:
+ Mat
+ Mat<T>
+ std::vector<T>
+ std::vector<std::vector<t>
+ std::vector<Mat>

Optional additional arguments are cv::noArray() to indicate no array
Class is for passing parameters

WarpMatrix is space to store motion model, depends on type of warp motion

motion Type depends on the kind of transformation
+ MOTION_TRANSLATION - first image shifted to second, only need to estimate 2 variables (X,Y)
+ MOTION_EUCLIDEAN - first image is rotated and shifted version of second, 3 variabes (X,Y, angle)
+ MOTION_AFFINE - combo of rotation, shift, scale and shear, transofrm has six parameters
+ MOTION_HOMOGRAPHY - 8 parameters, 3D effects 

TermCriteria is class to determine termination criteria for iterative algorithms
+ criteria can be type (type of terminationc iteria)
+ max count is number of iterations/elements to compute
+ epsilon is accuracy or change in parameters in which iterative algorithm stops

e.g., (InputArray templateImage, InputArray inputImage, InputOutputArray warpMatrix,
int motionType = MOTION_AFFINE, TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 50, 0.001), InputArray inputMask=noArray())

**/

double findTransformStreamlinedECC(InputArray templateImage,
                            InputArray inputImage,
                            InputOutputArray warpMatrix,
                            int motionType,
                            TermCriteria criteria)
{

		// set inputs to variables
    Mat src = templateImage.getMat();//template image
    Mat dst = inputImage.getMat(); //input image (to be warped)
    Mat map = warpMatrix.getMat(); //warp (transformation)

		/******************************* INPUT ERROR CHECKING ************************************/
    // Ensure inputs are not empty
		CV_Assert(!src.empty());
    CV_Assert(!dst.empty());

		// Ensure inputs are of the same type
    if( ! (src.type()==dst.type()))
        CV_Error( Error::StsUnmatchedFormats, "Both input images must have the same data type" );

    //accept only 1-channel images
    if( src.type() != CV_8UC1 && src.type()!= CV_32FC1)
        CV_Error( Error::StsUnsupportedFormat, "Images must have 8uC1 or 32fC1 type");

    if( map.type() != CV_32FC1)
        CV_Error( Error::StsUnsupportedFormat, "warpMatrix must be single-channel floating-point matrix");

		// Ensure warp matrix is either a 2x3 or a 3x3
    CV_Assert (map.cols == 3);
    CV_Assert (map.rows == 2 || map.rows ==3);

    CV_Assert (motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY ||
        motionType == MOTION_EUCLIDEAN || motionType == MOTION_TRANSLATION);

		/** Remove homography capabilities
    if (motionType == MOTION_HOMOGRAPHY){
        CV_Assert (map.rows ==3);
    }
		**/

    CV_Assert (criteria.type & TermCriteria::COUNT || criteria.type & TermCriteria::EPS);
    
		
		const int    numberOfIterations = (criteria.type & TermCriteria::COUNT) ? criteria.maxCount : 200;
    const double termination_eps    = (criteria.type & TermCriteria::EPS)   ? criteria.epsilon  :  -1;

    int paramTemp = 6;//default: affine
    
		/** Not needed for affine capabilities
		switch (motionType){
      case MOTION_TRANSLATION:
          paramTemp = 2;
          break;
      case MOTION_EUCLIDEAN:
          paramTemp = 3;
          break;
			// Remove homogrpahy 
      case MOTION_HOMOGRAPHY:
          paramTemp = 8;
          break;
		}
		**/


    const int numberOfParameters = paramTemp;

    const int ws = src.cols;
    const int hs = src.rows;
    const int wd = dst.cols;
    const int hd = dst.rows;

		// Create matrices to track X/Y coordinates and grid for warp
    Mat Xcoord = Mat(1, ws, CV_32F);
    Mat Ycoord = Mat(hs, 1, CV_32F);
    Mat Xgrid = Mat(hs, ws, CV_32F);
    Mat Ygrid = Mat(hs, ws, CV_32F);

		// Create array of float pointers 
    float* XcoPtr = Xcoord.ptr<float>(0);
    float* YcoPtr = Ycoord.ptr<float>(0);
    int j;
		
		// Set ptrs to width and height of source file
    for (j=0; j<ws; j++)
        XcoPtr[j] = (float) j;
    for (j=0; j<hs; j++)
        YcoPtr[j] = (float) j;
		
		// Fills output array with repeated copies of input array
		// (input matrix, flag for how many times src repeated along vertical, flag along horizontal, output matrix)
    repeat(Xcoord, hs, 1, Xgrid);
    repeat(Ycoord, 1, ws, Ygrid);

		// decrements reference count and frees matrix memory
    Xcoord.release();
    Ycoord.release();

		
    Mat templateZM    = Mat(hs, ws, CV_32F);// to store the (smoothed)zero-mean version of template
    Mat templateFloat = Mat(hs, ws, CV_32F);// to store the (smoothed) template
    Mat imageFloat    = Mat(hd, wd, CV_32F);// to store the (smoothed) input image
    Mat imageWarped   = Mat(hs, ws, CV_32F);// to store the warped zero-mean input image
    Mat imageMask		= Mat(hs, ws, CV_8U); //to store the final mask

    /** Not currently using inputMask
		Mat inputMaskMat = inputMask.getMat();
    

		//to use it for mask warping which holds values that dictate how much influence neighboring pixels have
		// on each new pixel value
    Mat preMask;
    if(inputMask.empty())
        preMask = Mat::ones(hd, wd, CV_8U);
    else
				// threshold separates image regions based on intensity variation between pixels
        threshold(inputMask, preMask, 0, 1, THRESH_BINARY);
		**/		
	
		// Set preMask matrix to be all 1s
		Mat preMask = Mat::ones(hd,wd,CV_8U);

		/** OPTIONAL
			Apply gaussian filtering to the premask, this 'blurs' the image to allow for higher level image structure
			detection
		**/
		src.convertTo(templateFloat, templateFloat.type());
    GaussianBlur(templateFloat, templateFloat, Size(5, 5), 0, 0);

    // Convert preMask to matrix of floating #
		Mat preMaskFloat;
    preMask.convertTo(preMaskFloat, CV_32F);
    
		// Apply Gaussian filter to preMaskFloat 
		// Gaussiablur(InputImage, OutputArray dst, Gaussian kernel Size, std dev in x, std dev in y)
		// Funtion blurs source image with specified Gaussian kernel	
		// Other filters exist for CUDA (e.g.,http://madsravn.dk/posts/simple-image-processing-with-cuda) 
		GaussianBlur(preMaskFloat, preMaskFloat, Size(5, 5), 0, 0);
   
	 	// Change threshold.
    preMaskFloat *= (0.5/0.95);
   
	 	// Rounding conversion
    preMaskFloat.convertTo(preMask, preMask.type());
    preMask.convertTo(preMaskFloat, preMaskFloat.type());

    dst.convertTo(imageFloat, imageFloat.type());
    
		// Blur source image 
		GaussianBlur(imageFloat, imageFloat, Size(5, 5), 0, 0);

    // Initialize matrices for gradients
    Mat gradientX = Mat::zeros(hd, wd, CV_32FC1);
    Mat gradientY = Mat::zeros(hd, wd, CV_32FC1);

		// Initialize dst matrices for the for loop
		Mat gradientXWarped = Mat(hs, ws, CV_32FC1);
    Mat gradientYWarped = Mat(hs, ws, CV_32FC1);

    // calculate first order image derivatives
    Matx13f dx(-0.5f, 0.0f, 0.5f);

		// convolves an image with the kernel - 
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

    
		/** Set flags to be used in warp affine transformations
			INTER_LINEAR: bilinear interpolation method
			WARP_INVERSE_MAP flag transforms source image using specified matrix: dst(x,y) = src(M11x + M12y + M12,21x
		 + M22y + M23) otherwise it is inverted and then put in formula instead of M which does not operate in place
		 **/
		const int imageFlags = INTER_LINEAR  + WARP_INVERSE_MAP;

		/** 
			INTER_NEAREST - nearest-neighbor interpolaion
			WARP_INVERSE_MAP - means M is inverse transformation (dst -> src)
		**/
    const int maskFlags  = INTER_NEAREST + WARP_INVERSE_MAP;

    // iteratively update map_matrix
    double rho      = -1;
    double last_rho = - termination_eps;
		// subbing 5 for numberOfIterations
    for (int i = 1; (i <= numberOfIterations) && (fabs(rho-last_rho)>= termination_eps); i++)
    {
			
				// warp-back portion of the inputImage and gradients to the coordinate space of the templateImage
        // warpAffine(InputArray src, OutputArray dst, InputArray M, Size dize, Flags, int borderMode, const Scalar)
				warpAffine(imageFloat, imageWarped,     map, imageWarped.size(),     imageFlags);
        warpAffine(gradientX,  gradientXWarped, map, gradientXWarped.size(), imageFlags);
				warpAffine(gradientY,  gradientYWarped, map, gradientYWarped.size(), imageFlags);
        warpAffine(preMask,    imageMask,       map, imageMask.size(),       maskFlags);

				/** Commented out because we are not using homography (first part of if statement is above)
				// warp-back portion of the inputImage and gradients to the coordinate space of the templateImage
        if (motionType != MOTION_HOMOGRAPHY)
        {
            warpAffine(imageFloat, imageWarped,     map, imageWarped.size(),     imageFlags);
            warpAffine(gradientX,  gradientXWarped, map, gradientXWarped.size(), imageFlags);
            warpAffine(gradientY,  gradientYWarped, map, gradientYWarped.size(), imageFlags);
            warpAffine(preMask,    imageMask,       map, imageMask.size(),       maskFlags);
        }
        else
        {
            warpPerspective(imageFloat, imageWarped,     map, imageWarped.size(),     imageFlags);
            warpPerspective(gradientX,  gradientXWarped, map, gradientXWarped.size(), imageFlags);
            warpPerspective(gradientY,  gradientYWarped, map, gradientYWarped.size(), imageFlags);
            warpPerspective(preMask,    imageMask,       map, imageMask.size(),       maskFlags);
        }

				**/

        // Calculate mean and std dev. of warped image matrix with image mask
				// Calculate mean and std dev. of smoothed image (with or without gaussian blur) with image mask
				
				Scalar imgMean, imgStd, tmpMean, tmpStd;
        meanStdDev(imageWarped,   imgMean, imgStd, imageMask);
        meanStdDev(templateFloat, tmpMean, tmpStd, imageMask);

				std::cout << "Template Mean is: " << tmpMean << std::endl << "Template std is: " << tmpStd << std:: endl;
				std::cout << "Image Mean is: " << imgMean << std::endl << "Image std is: " << imgStd << std:: endl;
		
				// Set zero mean image
        subtract(imageWarped,   imgMean, imageWarped, imageMask);//zero-mean input
				templateZM = Mat::zeros(templateZM.rows, templateZM.cols, templateZM.type());
        subtract(templateFloat, tmpMean, templateZM,  imageMask);//zero-mean template

        // Normalize template and image  
				const double tmpNorm = std::sqrt(countNonZero(imageMask)*(tmpStd.val[0])*(tmpStd.val[0]));
        const double imgNorm = std::sqrt(countNonZero(imageMask)*(imgStd.val[0])*(imgStd.val[0]));

				std::cout << "imgNorm " << imgNorm << std::endl;
				std::cout << "tmpNorm " << tmpNorm << std::endl;


        // calculate jacobian of image wrt parameters
        image_jacobian_affine_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, jacobian);
        

				/** Commented out because not using other types of transforms
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
				**/

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


        // calculate the parameter lambda to account for illumination variation
        imageProjectionHessian = hessianInv*imageProjection;
        const double lambda_n = (imgNorm*imgNorm) - imageProjection.dot(imageProjectionHessian);
        const double lambda_d = correlation - templateProjection.dot(imageProjectionHessian);
        if (lambda_d <= 0.0)
        {
            rho = -1;
            CV_Error(Error::StsNoConv, "The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped");

        }
        const double lambda = (lambda_n/lambda_d);

        // estimate the update step delta_p
        error = lambda*templateZM - imageWarped;
        project_onto_jacobian_ECC(jacobian, error, errorProjection);
        deltaP = hessianInv * errorProjection;

        // update warping matrix
        update_warping_matrix_ECC( map, deltaP, motionType);
    }

    // return final correlation coefficient
    return rho;
}
/* End of file. */

