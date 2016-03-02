//BASED ON:
//http://opencvexamples.blogspot.com/2014/06/discrete-fourier-transform.html

//GPU based:
//http://answers.opencv.org/question/11485/opencv-gpudft-distorted-image-after-inverse-transform/


#include "opencv2/highgui/highgui.hpp"
#include <iostream>
 
using namespace std;
using namespace cv;
 
int main()
{
    // Read image from file
    // Make sure that the image is in grayscale
    Mat img = imread("lenna.jpg",0);
     
    Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};
    Mat complexI;    //Complex plane to contain the DFT coefficients {[0]-Real,[1]-Img}
    merge(planes, 2, complexI);
    //dft(complexI, complexI);  // Applying DFT
    gpu::dft(complexI, complexI, coplexI.Size,);  // Applying DFT
    //NEED GpuMat for this!
 
    // Reconstructing original imae from the DFT coefficients
    Mat invDFT, invDFTcvt;
    idft(complexI, invDFT, DFT_SCALE | DFT_REAL_OUTPUT ); // Applying IDFT
    invDFT.convertTo(invDFTcvt, CV_8U); 
    imshow("Output", invDFTcvt);
 
    //show the image
    imshow("Original Image", img);
     
    // Wait until user press some key
    waitKey(0);
    return 0;
}


//fft2


//ifft2
