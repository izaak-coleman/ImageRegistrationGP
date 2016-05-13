#include <fstream>
#include "opencv2/opencv.hpp"
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <sys/time.h>
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

//#include <cublas_v2.h>
using namespace cv;
using namespace std;

/*
 Simple program to multiply all pixels in an 
 image by a multiplier scalar. 
*/

int main(){

    Mat lena;
    panda = imread("panda.jpg", CV_8U);
    
    cuda::GpuMat gpuLena;

    // upload to GPU
    std::clock_t upload_overhead;
    upload_overhead = std::clock();
    gpuPanda.upload(panda);
    std::cout << "upload time taken = " << (std::clock() - upload_overhead)/(double)CLOCKS_PER_SEC << std::endl;	

    // create GPU Mat object
    cuda::GpuMat gpuResult;
    Scalar multiplier = 1.5;

    // perform scalar matrix multiplication
    std::clock_t function;
    function = std::clock();
    cuda::multiply(gpuPanda, multiplier, gpuResult); 
    std::cout << "function time taken = " << (std::clock() - function)/(double)CLOCKS_PER_SEC << std::endl;	
	
    // create local Mat object to stor result
    Mat result;

    // download result
    std::clock_t download_overhead;
    download_overhead = std::clock();
    gpuResult.download(result);
    std::cout << "download time taken = " << (std::clock() - download_overhead)/(double)CLOCKS_PER_SEC << std::endl;	
    
    // write result to file
    imwrite("1.5xPanda.jpg", result);

    return 0;
}
