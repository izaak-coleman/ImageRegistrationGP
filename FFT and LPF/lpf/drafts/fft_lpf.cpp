// Jukka Soikkeli, 6 March 2016
// Samuel Martinet, 30 March 2016

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

////////////////////////////////////////////////////////////////////////////////
/* LPF on the CPU*/
void lpf(Mat &dft, const float R){

	Mat temp = Mat(dft.rows, dft.cols, CV_32F);
	Point centre = Point(dft.rows/2, dft.cols/2);

  float sqDistance;
	for(int i = 0; i < dft.rows; i++)
	{
		for(int j = 0; j < dft.cols; j++)
		{
		     sqDistance = (float) pow((i - centre.x), 2.0) + pow((double) (j - centre.y), 2.0);
		     if (sqDistance < pow(R, 2.0)){
      		        temp.at<float>(i,j) = dft.at<float>(i,j);
		     } else {
                  temp.at<float>(i,j) = 0;
         }
	  }
	}
  Mat planes[] = {temp, temp};
	merge(planes, 2, dft);
}

/* The following is taken verbatim from :
http://breckon.eu/toby/teaching/dip/opencv/lecture_demos/

// return a floating point spectrum magnitude image scaled for user viewing
// complexImg- input dft (2 channel floating point, Real + Imaginary fourier image)
// rearrange - perform rearrangement of DFT quadrants if true
// return value - pointer to output spectrum magnitude image scaled for user viewing
*/
Mat create_spectrum_magnitude_display(Mat& complexImg, bool rearrange){
    Mat planes[2];

    // compute magnitude spectrum (N.B. for display)
    // compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))

    split(complexImg, planes);
    magnitude(planes[0], planes[1], planes[0]);

    Mat mag = (planes[0]).clone();
    mag += Scalar::all(1);
    log(mag, mag);

    if (rearrange)
    {
        // re-arrange the quaderants
        shiftDFT(mag);
    }

    normalize(mag, mag, 0, 1, CV_MINMAX);

    return mag;
}

/* The following is taken verbatim from :
http://breckon.eu/toby/teaching/dip/opencv/lecture_demos/
// Rearrange the quadrants of a Fourier image so that the origin is at
// the image center
*/
void shiftDFT(Mat& fImage )
{
  Mat tmp, q0, q1, q2, q3;

	// first crop the image, if it has an odd number of rows or columns

	fImage = fImage(Rect(0, 0, fImage.cols & -2, fImage.rows & -2));

	int cx = fImage.cols/2;
	int cy = fImage.rows/2;

	// rearrange the quadrants of Fourier image
	// so that the origin is at the image center

	q0 = fImage(Rect(0, 0, cx, cy));
	q1 = fImage(Rect(cx, 0, cx, cy));
	q2 = fImage(Rect(0, cy, cx, cy));
	q3 = fImage(Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}


////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {

  if(argc!=2) {
    cout << "Incorrect number of arguments - one argument required for cpu/gpu choice." << endl;
    cout << "Usage: ./fft_lpf cpu   OR   ./fft_lpf gpu" << endl;
    return 1;
  }
  std:: string argv1 = argv[1];
  if(argv1!="cpu" && argv1!="gpu") {
    cout << "Incorrect command line argument (has to be 'cpu' or 'gpu')!" << endl;
    cout << "Usage: ./fft_lpf cpu   OR   ./fft_lpf gpu" << endl;
    return 1;
  }


  //-------------- Preparation ------------------------------
  // Read image from file, making sure that the image is in grayscale
  Mat img = imread("lenna.jpg",CV_LOAD_IMAGE_GRAYSCALE);

  // Make space for complex components
  Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)}; //or CV_32FC1
  Mat complexI;    //Complex plane to contain the DFT coefficients
  merge(planes, 2, complexI);  //creates a two-channel complexI array from the planes


  // ---- CPU version ----
  if(argv1=="cpu") {
    // Applying forward DFT (in place)
    dft(complexI, complexI);

    // LPF
		Mat lpf;
		lpFilter = complexI.clone();
    lpf(lpFilter,500);
		shiftDFT(complexI);
    mulSpectrums(complexI, filter, complexI, 0);
		shiftDFT(complexI);
		mag = create_spectrum_magnitude_display(complexI, true);


		Mat out, outimg; //make matrices

// COPY MODE ON
const string spectrumMagName = "Magnitude Image (log transformed)"; // window name
const string lowPassName = "Butterworth Low Pass Filtered (grayscale)"; // window name
const string filterName = "Filter Image"; // window nam
Mat filterOutput, imgOutput;
// do inverse DFT on filtered image
idft(complexI, complexI);

// split into planes and extract plane 0 as output image
split(complexI, planes);
normalize(planes[0], imgOutput, 0, 1, CV_MINMAX);

// do the same with the filter image
split(lpFilter, planes);
normalize(planes[0], filterOutput, 0, 1, CV_MINMAX);

// display image in window
imshow(spectrumMagName, mag);
imshow(lowPassName, imgOutput);
imshow(filterName, filterOutput);
// COPY MODE OFF

/*

    // Inverse DFT
    Mat out, outimg; //make matrices
    dft(complexI,out, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT); //getting inverse using dft function
    out.convertTo(outimg, CV_8U);
    imshow("Output", outimg);
    imwrite( "out.jpg", outimg );

		// LPF display
*/
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


    /* ODD RESULT - probalbly due to the complex-complex - complex-real setting of the DFT functions...
       to be debugged, with the help of:
       http://answers.opencv.org/question/11485/opencv-gpudft-distorted-image-after-inverse-transform/
       http://docs.opencv.org/2.4/modules/gpu/doc/image_processing.html#gpu-dft
    */

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
