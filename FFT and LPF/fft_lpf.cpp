//BASED ON:
//http://opencvexamples.blogspot.com/2014/06/discrete-fourier-transform.html

//GPU based:
//http://answers.opencv.org/question/11485/opencv-gpudft-distorted-image-after-inverse-transform/

//#include <opencv2/core/core.hpp>      // Basic OpenCV structures
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <opencv2/gpu/gpu.hpp>        // GPU structures and methods
 
using namespace std;
using namespace cv;
 
int main()
{
    // Read image from file
    // Make sure that the image is in grayscale
    Mat img = imread("lenna.jpg",0);

    //-------------- CPU version ------------------------------ 
    /*
    Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};
    Mat complexI;    //Complex plane to contain the DFT coefficients {[0]-Real,[1]-Img}
    merge(planes, 2, complexI);
    dft(complexI, complexI);  // Applying DFT
    */

    // using https://github.com/Itseez/opencv/blob/master/samples/gpu/performance/tests.cpp
    Mat src, dst;
    src = img;
    dft(src,dst);


    //-------------- GPU version ------------------------------
    gpu::gpuMat gpuimg;
    gpu::gpuMat gpudst;
    gpuimg.upload(img);  //converting image mat to gpu mat
    gpu::dft(gpuimg, gpudst, gpuimg.Size);  // Applying DFT, using real images
    //gpu::dft(complexI, complexI, coplexI.Size,);  // Applying DFT
    //NEED GpuMat for this!
 



    //------ CPU version --------
    // Reconstructing original imae from the DFT coefficients
    /*
    Mat invDFT, invDFTcvt;

    idft(complexI, invDFT, DFT_SCALE | DFT_REAL_OUTPUT ); // Applying IDFT
    invDFT.convertTo(invDFTcvt, CV_8U); 
    imshow("Output", invDFTcvt);


    //show the image
    imshow("Original Image", img);


    */


    //OWN VERSION

       //CPU
       Mat out, outimg; //make matrices
       dft(dst,out, DFT_INVERSE | DFT_REAL_OUTPUT); //getting inverse using dft function
       out.convertTo(outimg, CV_8U);
       //imshow("Output", outimg);
       imwrite( "out.jpg", gray_image );

       //GPU
       gpu::gpuMat gpuout, gpuoutimg;
       gpu::dft(dst,gpuout, DFT_INVERSE | DFT_REAL_OUTPUT);
       gpuout.convertTo(gpuoutimg, CV_8U);
       //imshow("Output (GPU)", gpuoutimg);
       imwrite( "outgpu.jpg", gray_image );
    //show the image
       //    imshow("Original Image", img);

    // Wait until user press some key
    //waitKey(0);
    return 0;
}


//fft2


//ifft2
