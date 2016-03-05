//BASED ON:
//http://opencvexamples.blogspot.com/2014/06/discrete-fourier-transform.html
//http://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html

//GPU based:
//http://answers.opencv.org/question/11485/opencv-gpudft-distorted-image-after-inverse-transform/

/*
#include <opencv2/core/core.hpp>      // Basic OpenCV structures
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

 

//#include <opencv2/gpu/gpu.hpp>        // GPU structures and methods

#include <fstream>
#include "opencv2/opencv.hpp"
*/

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
 
int main() {
  // Read image from file
  // Make sure that the image is in grayscale
  //Mat img = imread("lenna.jpg",0);
  Mat img = imread("lenna.jpg",CV_LOAD_IMAGE_GRAYSCALE);
  
  //-------------- CPU version ------------------------------ 
  //Make space for complex components
  //Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};
  Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32FC1)};
  Mat complexI;    //Complex plane to contain the DFT coefficients {[0]-Real,[1]-Img}
  merge(planes, 2, complexI);  //creates a two-channel complexI array from the planes
  // Applying DFT (in place)
  // dft(complexI, complexI);
  
  /* If want to view the fourier transform "image", add here things from 
     http://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
     These seem to be unnecessary for the basic case
  */
  

    //-------------- GPU version ------------------------------
    cuda::GpuMat complexIgpu;
    cuda::GpuMat gpuimg;
    gpuimg.upload(img);  //converting image mat to gpu mat
    complexIgpu.upload(complexI);  //converting image mat to gpu mat
    //cuda::dft(gpuimg, complexIgpu, complexIgpu.size());  // Applying DFT, using complex
    cuda::dft(complexIgpu, complexIgpu, complexIgpu.size());  // Applying DFT, using complex
    //cuda::dft(complexIgpu, gpudst, complexIgpu.size());  // Applying DFT, using real images


    //OWN VERSION
  /*
       //CPU
       Mat out, outimg; //make matrices
       dft(complexI,out, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT); //getting inverse using dft function
       out.convertTo(outimg, CV_8U);
       imshow("Output", outimg);
       //imwrite( "out.jpg", outimg );
       */

       //GPU
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


    /* ODD RESULT - probalbly due to the complex-complex - complex-real thing...
       debug using:
       http://answers.opencv.org/question/11485/opencv-gpudft-distorted-image-after-inverse-transform/
       http://docs.opencv.org/2.4/modules/gpu/doc/image_processing.html#gpu-dft
    */
    
    //show the image
    imshow("Original Image", img);
  
    // Wait until user press some key
    waitKey(0);
    return 0;
}


//fft2


//ifft2
