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

using namespace cv;
using namespace std;

void gpuDotProduct(cuda::GpuMat src1, cuda::GpuMat src2, cuda::GpuMat dest);

int main(){
  
    Mat a = (Mat_<double>(5,5) << 234, 233, 126, 211, 185, 121, 198, 178, 216, 
		231, 234, 233, 126, 211, 185, 121, 198, 178, 216, 231, 232, 213, 186, 251,
		135);
    Mat b = (Mat_<double>(5,5) << 232, 213, 186, 251, 135, 221, 190, 128, 236, 
		231, 234, 233, 126, 211, 185, 121, 198, 178, 216, 231, 232, 213, 186, 251,
		135);
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

		cuda::GpuMat input;


		gpuDotProduct(gpua, gpub, input);
  
    cout << "sums" << endl;
    cout << "multiply" << multotal << endl;
    cout << "cpu" << total << endl;

    cout << ".dot" << a.dot(b) << endl;
    return 0;
}

void gpuDotProduct(cuda::GpuMat src1, cuda::GpuMat src2, cuda::GpuMat fold){

  // perform elementwise multiplication
  cuda::multiply(src1, src2, fold);

 /* ------------------------ Cols folding -----------------------------------*/
  // sum elements of fold
  int col_lastResult=1, col_currentResult=1;
  for(int i=1; col_currentResult < fold.cols; i++){
    col_lastResult = pow(2, i-1);			// last whole number log2
    col_currentResult = pow(2, i);			// current whole number log2
  }

  // add offset to remainder of fold, to make log of fold.cols a whole number
  int col_distanceFromLog2End = src1.cols - col_lastResult;		// offset from log2
	
  cuda::GpuMat col_offcut, col_destSection;
  col_offcut = fold.colRange(col_lastResult, fold.cols);
  col_destSection = fold.colRange((col_lastResult - col_distanceFromLog2End), col_lastResult);
  cuda::add(col_destSection, col_offcut, col_destSection);


  fold = fold.colRange(0, col_lastResult);		// update the new size of dest

	// fold!
  while(fold.cols != 1) {

  cuda::GpuMat rightHalf, leftHalf;
    // split into right and left halfs
    rightHalf = fold.colRange(0, (fold.cols/2));
    leftHalf = fold.colRange((fold.cols/2), (fold.cols));
  
    // add halfs together
    cuda::add(rightHalf, leftHalf, fold);
    cout << "fold cols reduction" << fold.cols << endl;
 }


 /* ------------------------ Rows folding -----------------------------------*/

  // sum elements of fold
  int row_lastResult=1, row_currentResult=1;
  for(int i=1; row_currentResult < fold.rows; i++){
    row_lastResult = pow(2, i-1);			// last whole number log2
    row_currentResult= pow(2, i);			// current whole number log2
  }

  // add offset to remainder of fold, to make log of fold.cols a whole number
  int row_distanceFromLog2End = src1.rows - row_lastResult;		// offset from log2
  cuda::GpuMat row_offcut, row_destSection;
  row_offcut = fold.rowRange(row_lastResult, fold.rows);
  row_destSection = fold.rowRange((row_lastResult - row_distanceFromLog2End), row_lastResult);
  cuda::add(row_destSection, row_offcut, row_destSection);


  fold = fold.rowRange(0, row_lastResult);		// update the new size of fold

	// fold!
  while(fold.rows!= 1) {

  cuda::GpuMat rightHalf, leftHalf;
    // split into right and left halfs
    rightHalf = fold.rowRange(0, (fold.rows/2));
    leftHalf = fold.rowRange((fold.rows/2), (fold.rows));
  
    // add halfs together
    cuda::add(rightHalf, leftHalf, fold);
    cout << "fold row reduction" << fold.rows << endl;
 }

 Mat dest_download;
 fold.download(dest_download);

 cout << "gpuDotProduct dest: " << dest_download << endl;

}

