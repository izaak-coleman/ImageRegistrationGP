# ======== Sample CMakeLists.txt file for pocketReg project ============
# ----- Created by: Jukka Soikkeli (jes15@ic.ac.uk) on 2016-01-27 ------
cmake_minimum_required(VERSION 2.8)
project( displayImage ) #replace with project name: project( <proj_name> )

# ==================== Linking to openCV 3.1.0 =========================
#    (this part can be simly copy-pasted to other CMakeLists files)
set(OpenCV_DIR "/vol/bitbucket/ic711/opencv-3.1.0/release")
set(OpenCV_CONFIG_PATH "/vol/bitbucket/ic711/opencv-3.1.0/cmake")

find_package( OpenCV 3.1.0 EXACT CONFIG REQUIRED )  #Finds package config for OpenCV 3.1.0
message("OpenCV package version " ${OpenCV_VERSION}) #Echoes the package version
# ======================================================================


# ----- link libraries (CHANGE TO REFLECT EXECUTABLE) -----
add_executable( Display DisplayImage.cpp )  #change to reflect project 
target_link_libraries( Display ${OpenCV_LIBS}) #replace "DisplayImage"

# -- Template for this part --:
#add_executable( <executable> <source> )
#target_link_libraries( <executable> ${OpenCV_LIBS})

