cmake_minimum_required(VERSION 2.8)
project( image_alignment_modified )
# ----- Linking to openCV 3.1.0 -----
#THE FOLLOWING CONFIGURATION REQUIRED::
set(OpenCV_DIR "/vol/bitbucket/ic711/opencv-3.1.0/release")
set(OpenCV_CONFIG_PATH "/vol/bitbucket/ic711/opencv-3.1.0/cmake")
find_package( OpenCV 3.1.0 EXACT CONFIG REQUIRED )  #FINDS package config for OpenCV 3.1.0
message("OpenCV package version " ${OpenCV_VERSION})

# ===================== FLAGS (e.g. for gprof) ===================== 
# ---------------------------- GCC flags ---------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pg")     #-Wall and -pg added to existing
# --------------------------- Linker flags -------------------------
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")       #-pg added to existing
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg") #-pg added to existing
#The "-pg" flag used for gprof profiling; other flags can be added as above

# ----- link libraries -----
add_executable( ecc "image_alignment_modified.cpp" )
target_link_libraries( ecc ${OpenCV_LIBS})
