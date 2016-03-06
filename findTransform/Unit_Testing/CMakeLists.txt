cmake_minimum_required(VERSION 2.8)
project( findTransform )



find_package(Boost 1.54.0)
if(Boost_FOUND)
   include_directories(${Boost_INCLUDE_DIRS})
message("Boost package version " ${Boost_VERSION})


# ----- Linking to openCV 3.1.0 -----
#THE FOLLOWING CONFIGURATION REQUIRED::
set(OpenCV_DIR "/vol/bitbucket/ic711/opencv-3.1.0/release")
set(OpenCV_CONFIG_PATH "/vol/bitbucket/ic711/opencv-3.1.0/cmake")
find_package( OpenCV 3.1.0 EXACT CONFIG REQUIRED )  #FINDS package config for OpenCV 3.1.0
message("OpenCV package version " ${OpenCV_VERSION})
# ----- link libraries -----
add_executable( "unit_test" saveWarp.cpp test_saveWarp.cpp)
target_link_libraries( unit_test ${OpenCV_LIBS}${Boost_LIBS})


endif()

