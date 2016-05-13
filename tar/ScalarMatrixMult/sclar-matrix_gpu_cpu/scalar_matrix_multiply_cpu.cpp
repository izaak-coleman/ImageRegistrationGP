#include "opencv2/opencv.hpp"
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <sys/time.h>

//cuda include files 
using namespace cv;
using namespace std;

/*
 Simple program to multiply all pixels in an 
 image by a multiplier scalar. 
 */

int main(){

  // read local images into Mat objects
  Mat img, result; 
  double funcTime;
  img = imread("panda.jpg", CV_8U);             
  
  // set constand scalar multiplier
  Scalar multiplier = 1.5;

  // perform scalar matrix multiplication
  std::clock_t function;
  function = std::clock();                     
  multiply(img, multiplier, result);
  funcTime = (std::clock() - function)/(double)CLOCKS_PER_SEC; 
  std::cout << "function time taken = " << funcTime << std::endl;
  
  // write result to file
  imwrite("1.5xPanda.jpg", result);      // save result
  
  return 0;
}
