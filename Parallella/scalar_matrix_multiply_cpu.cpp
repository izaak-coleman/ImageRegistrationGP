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

    Mat img, result;
		double funcTime;
    img = imread("lenna.jpg", CV_8U);							// load data	
		Scalar multiplier = 1.5;
    std::clock_t function;


    function = std::clock();											// CLOCK START
    multiply(img, multiplier, result);
		funcTime = (std::clock() - function)/(double)CLOCKS_PER_SEC; // CLOCK END

    std::cout << "function time taken = " << funcTime << std::endl;
    imwrite("1.5xlena.jpg", result);			// save result

    return 0;
}
