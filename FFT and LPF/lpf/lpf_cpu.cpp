// 2nd of avril

#include "opencv2/opencv.hpp"
 
#include <iostream>

using namespace std;
using namespace cv;

/* LPF on the CPU*/
Mat lpf(Mat &dft, Mat img, const float R){

    Mat pl[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)}; 
    Mat temp;    
    merge(pl, 2, temp);  
    Point centre = Point(dft.rows/2, dft.cols/2);

    float sqDistance;
	for(int i = 0; i < dft.rows; i++)
        {
                for(int j = 0; j < dft.cols; j++)
                {
                     sqDistance = (float) pow((double)(i - centre.x), 2.0) + pow((double) (j - centre.y), 2.0);
                     //if (sqDistance <= pow(R, 2.0)){
                     if (sqDistance > pow(R, 2.0)){
                        temp.at<float>(i,j) = dft.at<float>(i,j);
                     } else {
                  temp.at<float>(i,j) = 0;
         }
          }
        }
    return temp;
}

int main(int argc, char *argv[]) {




  if(argc!=2) {
    cout << "Incorrect number of arguments - one argument required the parameter of the LPF." << endl;
    cout << "Usage: './lpf_cpu 75' for instance" << endl;
    return 1;
  }

 
	int argv1=  std::atoi(argv[1]);
	cout << "Parameter R of the LPF: " << argv1 << endl;

  // Read image from file, making sure that the image is in grayscale
  Mat img = imread("lenna.jpg",CV_LOAD_IMAGE_GRAYSCALE);

  // Make space for complex components
  Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)}; 
  Mat complexI;   
  merge(planes, 2, complexI);  

 // Applying forward DFT (in place)
  dft(complexI, complexI);

 // LPF
  Mat filter = lpf(complexI, img, argv1);

 // Inverse DFT
  Mat out, outimg;  
  dft(filter,out, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT);  
  out.convertTo(outimg, CV_8U);
 
  imshow("Output", outimg);
  imwrite( "out.jpg", outimg );
  imshow("Original Image", img);

  // Wait until user presses a key before exiting
  waitKey(0);  
  return 0;
}

 
