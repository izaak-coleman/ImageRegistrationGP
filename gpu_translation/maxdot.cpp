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

int main(){
  
	  Mat a = (Mat_<double>(3,3) << 234, 233, 126, 211, 185, 121, 198, 178, 216);
	  Mat b = (Mat_<double>(3,3) << 232, 213, 186, 251, 135, 221, 190, 128, 236);
		Mat mul;


		a.convertTo(a, CV_32F);
		b.convertTo(b, CV_32F);

		cuda::GpuMat gpua, gpub, gpumul;
		gpua.upload(a);
		gpub.upload(b);

		cuda::multiply(gpua, gpub, gpumul);
		gpumul.download(mul);

		double multotal=0;

		for(int j=0; j < mul.cols; j++){
			for(int i=0; i < mul.rows; i++){
				multotal += mul.at<float>(i,j);
			}
		}
////////////////////////////////////////////// Col compress

		cuda::GpuMat lastColumn, lastRow;
		bool oddrow = false, oddcol = false;
		// if odd cols store final column
		if(gpumul.cols % 2 == 1){
			lastColumn = gpumul.col(gpumul.cols-1);
			oddcol = true;
		}
		cout << "gpumul" << gpumul.cols << endl;

		cuda::GpuMat fold;

		// crop an even matrix for folding
		if(oddcol){
			fold = gpumul.colRange(0, gpumul.cols-1);
			Mat coutFold;
			fold.download(coutFold);
			cout << "fold matrix: " << coutFold << endl;
		}
			cout << "fold " << fold.cols << endl;
		
		while(fold.cols != 1) {

			cuda::GpuMat rightHalf, leftHalf;
			// split into right and left halfs
			rightHalf = fold.colRange(0, (fold.cols/2));
			leftHalf = fold.colRange((fold.cols/2), (fold.cols));
	
			// add halfs together
			cuda::add(rightHalf, leftHalf, fold);
			cout << "fold reduction" << fold.cols << endl;
		}

		// if odd number of columns, add the odd column
		if(oddcol){
			cuda::add(fold, lastColumn, fold);
		}

///////////////////////////////////// row compress

		// if odd rows store final rows	
		if(fold.rows % 2 == 1){
			lastRow = fold.row(fold.rows-1);
			oddrow = true;
		}


		// crop an even matrix for folding
		if(oddrow){
			fold = fold.rowRange(0, fold.rows-1);
		}

		
		while(fold.rows !=1) {
			cuda::GpuMat topHalf, bottomHalf;

			topHalf = fold.rowRange(0, (fold.rows/2));
			bottomHalf = fold.rowRange(fold.rows/2, (fold.rows));

			// add halfs together
			cuda::add(topHalf, bottomHalf, fold);
		}

		// if odd rows, add last row element
		if(oddrow){
			cuda::add(fold, lastRow, fold);
		}

		Mat gpuDotResult;
		fold.download(gpuDotResult);
		cout <<  "gpuDOtReulst: " << gpuDotResult << endl;








		double total = 0;
		double elem = 0;

		for(int j=0; j < a.cols; j++){
			for(int i=0; i < b.rows; i++){
				elem = a.at<float>(i,j) * b.at<float>(i,j);
				total += elem;
			}
		}
	
		cout << "sums" << endl;
		cout << "multiply" << multotal << endl;
		cout << "cpu" << total << endl;

		cout << ".dot" << a.dot(b) << endl;
		return 0;
}
