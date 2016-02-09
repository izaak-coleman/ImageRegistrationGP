//#include <iostream>
//#include "opencv2/opencv.hpp"
//#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include "opencv2/opencv.hpp"
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

int main (int argc, char* argv[])
{
    try
    {
        cv::Mat src_host = cv::imread("castle.jpg", CV_LOAD_IMAGE_GRAYSCALE);
        cv::cuda::GpuMat dst, src, xmap, ymap;
        src.upload(src_host);
    
        cv::Mat M = (cv::Mat_<double>(2,3) << 1.0, 0.0, 50.0, 0.0, 1.0, 50.0);
//        cv::cuda::buildWarpAffineMaps(M, false, src_host.size(), xmap, ymap);

        cv::cuda::warpAffine(src, dst, M, src_host.size());


        cv::Mat result_host;
        dst.download(result_host);
        cv::imshow("Result", result_host);
        cv::waitKey();
    }
    catch(const cv::Exception& ex)
    {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    return 0;
}
