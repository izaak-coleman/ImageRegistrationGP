#include <fstream>
#include "opencv2/opencv.hpp"
#include <cstdlib>
#include <cmath>
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

#include <cublas_v2.h>
using namespace cv;
using namespace std;


int main(int argc, char** argv){
	Mat srcimgA = imread("castle.jpg", CV_32F);
	Mat srcimgB = imread("480x360_castle_affine.jpg", CV_32F);

	Mat imgA, imgB;
	cvtColor(srcimgA, imgA, CV_BGR2GRAY);
	cvtColor(srcimgB, imgB, CV_BGR2GRAY);




	imgA.convertTo(imgA, CV_32F);
	imgB.convertTo(imgB, CV_32F);

	if(imgA.isContinuous() == true){
		cout << "img a cont" << endl;
	}
	if(imgB.isContinuous() == true){
		cout << "img b cont" << endl;
	}

	size_t lenA = imgA.total() * imgA.channels();
	size_t lenB = imgB.total() * imgB.channels();

	int lenA0 = lenA & -4;

	int blockSize0 = (1 << 13);
	cout << "blockSize0: " << blockSize0 << endl;
	cout << "bit change " << lenA0 << endl;

	cout << "imgA.total(): " << imgA.total() << endl;
	cout << "imgB.total(): " << imgB.total() << endl;

	cout << "lenA: " << lenA << endl;
	cout << "lenB: " << lenB << endl;

	cout << "imgA.channels(): " <<  imgA.channels() << endl;

	if(lenA == (size_t)(int)lenA){
		cout << "type casts equivalent" << endl;
	}



	return 0;
}
