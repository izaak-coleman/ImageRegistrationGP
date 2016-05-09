#include <iostream>
#include "opencv2/opencv.hpp"

#include <cstring>
/*
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudastereo.hpp"
#include "opencv2/cudawarping.hpp"*/

using namespace cv;

int main (int argc, char* argv[])
{
//cuda::GpuMat src;
    Mat src_host;
    // Usage: <cmd> <file_in> <file_out>
    // Read original image

    src_host = imread(argv[1], IMREAD_UNCHANGED); // read in img
		//src.upload(src_host);
    cuda::GpuMat dest;
    Mat temp_dest = Mat::zeros(src.rows, src.cols, CV_32FC3); // size of dst
    dest.upload(temp_dest);
    Mat dx  = (Mat_<double>(3,3) << 0, 2, 0, 2, 5, 2, 0, 2, 0);

    cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createLinearFilter(src.type(), dest.type(), dx);
    filter->apply(src, dest);

    Mat result_src, result_dest;
    src.download(result_src);
    dest.download(result_dest);

    namedWindow(argv[1], WINDOW_AUTOSIZE);
    namedWindow("After filter", WINDOW_AUTOSIZE);

    imshow(argv[1], src);
    imshow("After filter", dest);
    waitKey();
    return 0;
}

