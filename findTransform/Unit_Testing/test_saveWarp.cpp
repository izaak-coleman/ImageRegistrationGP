#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>
#include "opencv2/opencv.hpp"
#include "saveWarp.h"
#include <sstream>

BOOST_AUTO_TEST_CASE(saveWarp_test)
{
    std::string testFileName;
    cv::Mat testWarp;
    int testMotionType;

    int output = saveWarp(testFileName, testWarp, testMotionType);

    //TEST1: 

    BOOST_REQUIRE_EQUAL(0, output);
}
