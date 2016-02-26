cmake_minimum_required(VERSION 2.8)
project( sorting )

# ===================== FLAGS (e.g. for gprof) ===================== 
# ---------------------------- GCC flags ---------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pg")     #-Wall and -pg added to existing
# --------------------------- Linker flags -------------------------
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")       #-pg added to existing
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg") #-pg added to existing
#The "-pg" flag used for gprof profiling, "-Wall" is for warnings
#Other flags can be added as above (although may need to check if for compilation or linker etc.)


add_executable( sorting sorting.cpp )
