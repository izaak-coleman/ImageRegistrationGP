cmake_minimum_required(VERSION 2.8)
project( emirreg ) #replace "emirreg" with project name: project( <proj_name> )

# ==================== Linking to openCV 3.1.0 =========================
set(OpenCV_DIR "/vol/bitbucket/ic711/opencv-3.1.0/release")
set(OpenCV_CONFIG_PATH "/vol/bitbucket/ic711/opencv-3.1.0/cmake")

find_package( OpenCV 3.1.0 EXACT CONFIG REQUIRED )   #Finds package config for OpenCV 3.1.0
message("OpenCV package version " ${OpenCV_VERSION}) #Echoes the package version
# ======================================================================


# ===================== FLAGS (e.g. for gprof) ===================== 
# ---------------------------- GCC flags ---------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pg")     #-Wall and -pg added to existing
# --------------------------- Linker flags -------------------------
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")       #-pg added to existing
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg") #-pg added to existing
#The "-pg" flag used for gprof profiling, "-Wall" is for warnings
#Other flags can be added as above (although may need to check if for compilation or linker etc.)


# ----- link libraries (CHANGE TO REFLECT EXECUTABLE) -----
add_executable( emir ima.cpp )  #replace "emir" and "ima.cpp"
target_link_libraries( emir ${OpenCV_LIBS}) #replace "executable"

