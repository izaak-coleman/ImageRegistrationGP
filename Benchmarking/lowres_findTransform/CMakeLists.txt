cmake_minimum_required(VERSION 2.8)
project( findTransform )
# ----- Linking to openCV 3.1.0 -----
#THE FOLLOWING CONFIGURATION REQUIRED::
set(OpenCV_DIR "/vol/bitbucket/ic711/opencv-3.1.0/release")
set(OpenCV_CONFIG_PATH "/vol/bitbucket/ic711/opencv-3.1.0/cmake")
find_package( OpenCV 3.1.0 EXACT CONFIG REQUIRED )  #FINDS package config for OpenCV 3.1.0
message("OpenCV package version " ${OpenCV_VERSION})
# ----- link libraries -----
add_executable( lowres_findTransform_V2 "lowres_findTransform_V2.cpp" )
target_link_libraries( lowres_findTransform_V2 ${OpenCV_LIBS})
