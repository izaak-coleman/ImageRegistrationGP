// Jukka Soikkeli, 6 March 2016 

//#include <opencv2/core/core.hpp>      // Basic OpenCV structures
#include "opencv2/opencv.hpp"
//#include "opencv2/highgui/highgui.hpp"
#include <iostream>

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

using namespace std;
using namespace cv;
 
int main(int argc, char *argv[]) {
 
  if(argc!=2) {
    cout << "Incorrect number of arguments - one argument required for cpu/gpu choice." << endl;
    cout << "Usage: ./lpf_only cpu   OR   ./fft_only gpu" << endl;
    return 1;
  }
  std:: string argv1 = argv[1];
  if(argv1!="cpu" && argv1!="gpu") {
    cout << "Incorrect command line argument (has to be 'cpu' or 'gpu')!" << endl;    
    cout << "Usage: ./fft_only cpu   OR   ./fft_only gpu" << endl;
    return 1;
  }


  //-------------- Preparation ------------------------------ 
  // Read image from file, making sure that the image is in grayscale
  Mat img = imread("lenna.jpg",CV_LOAD_IMAGE_GRAYSCALE);
  Mat out;

  Mat kernel;
  Point anchor;
  double delta;
  int ddepth;
  int kernel_size;

  /// Initialize arguments for the filter
  anchor = Point( -1, -1 );
  delta = 0;
  ddepth = -1;  // CV_32F, but when negative, set to source depth;

  /// Update kernel size for a normalized box filter

  /*
  kernel_size = 3;// + 2*( ind%5 );
  //kernel_size = 5;// + 2*( ind%5 );
  kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
  */
  kernel = 1/36.0*(Mat_<double>(3,3) << 1, 4, 1, 4, 16, 4, 1, 4, 1);

float recipro_2_by[16] = {
		1.000000000000000, // 1/2^0
		0.500000000000000, // 1/2^1
		0.250000000000000, // 1/2^2
		0.125000000000000, // 1/2^3
		0.062500000000000, // 1/2^4
		0.031250000000000, // 1/2^5
		0.015625000000000, // 1/2^6
		0.007812500000000, // 1/2^7
		0.003906250000000, // 1/2^8
		0.001953125000000, // 1/2^9
		0.000976562500000, // 1/2^10
		0.000488281250000, // 1/2^11
		0.000244140625000, // 1/2^12
		0.000122070312500, // 1/2^13
		0.000061035156250, // 1/2^14
		0.000030517578125, // 1/2^15
};

 kernel = Mat(4,4,CV_32F,recipro_2_by);


  filter2D(img,out,ddepth,kernel);

  //Save the output image
  imwrite( "outLPF.jpg", out );

  // Show the image
  imshow("Original Image", img);
  imshow("Output Image", out);
  
  // Wait until user presses a key before exiting
  waitKey(0); // use while images shown, take out if only image saving is required
  return 0;
}




/*

	   ///FFT STUFF

  // Make space for complex components
  Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)}; //or CV_32FC1
  Mat complexI;    //Complex plane to contain the DFT coefficients
  merge(planes, 2, complexI);  //creates a two-channel complexI array from the planes


  // ---- CPU version ----
  if(argv1=="cpu") {
    // Applying forward DFT (in place)
    dft(complexI, complexI);
    

    // LPF GOES HERE
    
    
    // Inverse DFT
    Mat out, outimg; //make matrices
    dft(complexI,out, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT); //getting inverse using dft function
    out.convertTo(outimg, CV_8U);
    imshow("Output", outimg);
    imwrite( "out.jpg", outimg );
  }


  //-------------- GPU version ------------------------------
  //////////////////////////////////////
  // WORK IN PROGRESS - SOME BUG REMAINS
  //////////////////////////////////////
  if(argv1=="gpu") {
    // --- Preparation ---
    cuda::GpuMat complexIgpu;
    cuda::GpuMat gpuimg;
    gpuimg.upload(img);  //converting image mat to gpu mat
    complexIgpu.upload(complexI);  //converting image mat to gpu mat

    // Applying forward DFT
    //cuda::dft(gpuimg, complexIgpu, complexIgpu.size());  // Applying DFT, using complex
    cuda::dft(complexIgpu, complexIgpu, complexIgpu.size());  // Applying DFT, using complex
    //cuda::dft(complexIgpu, gpudst, complexIgpu.size());  // Applying DFT, using real images


    // LPF GOES HERE


    // Inverse DFT
    cuda::GpuMat gpuout;//, gpuoutimg;
    cuda::dft(complexIgpu,gpuout, complexIgpu.size(), DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
    //cuda::dft(complexIgpu,gpuout, complexIgpu.size(), DFT_INVERSE);

    Mat out(gpuout);
    Mat outimg;
    //double n,x;
    //minMaxIdx(gpuout, &n, &x);
    //gpuout.convertTo(outimg, CV_8U, 255.0/x);
    
    out.convertTo(outimg, CV_8U);
    // imshow("Output (GPU)", gpuoutimg);
    imshow("Output (GPU)", outimg);
    //imwrite( "outgpu.jpg", gpuoutimg );
    imwrite( "outgpu.jpg", outimg );


    // ODD RESULT - probalbly due to the complex-complex - complex-real setting of the DFT functions...
    //   to be debugged, with the help of:
    //   http://answers.opencv.org/question/11485/opencv-gpudft-distorted-image-after-inverse-transform/
    //   http://docs.opencv.org/2.4/modules/gpu/doc/image_processing.html#gpu-dft
    //
    
  }


  // Show the image
  imshow("Original Image", img);
  
  // Wait until user presses a key before exiting
  waitKey(0); // use while images shown, take out if only image saving is required
  return 0;
}


// Helpful websites:
//http://opencvexamples.blogspot.com/2014/06/discrete-fourier-transform.html
//http://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html

//GPU based:
//http://answers.opencv.org/question/11485/opencv-gpudft-distorted-image-after-inverse-transform/
*/
