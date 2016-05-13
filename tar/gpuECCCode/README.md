### gpuECC: GPU conversion V3 of findTransformECC from OpenCV3.0

Version compiled with Nvidia Compiler (nvcc) from CUDA Toolkit 6.5.14

OpenCV binaries included from non standard path: /vol/bitbucket/ic711/usr/local/

Ensure path is included in Makefile before compilation

Subdirectories:
  * withoutTimers: Contains gpu ECC program without benchmarking timers
  * gauss:          Contains gpu ECC program with gauss optimization 
                   and timed functions
  * nongauss:      Contains gpu ECC program without gauss opt. and timed
                   functinos - equivalent to withoutTimers exe but timed

Makefiles all reference non standard path above for linking 
Generated executable requires commandline inputs in order
1) template image 2) deformed image (will be registered) 3) f.name for warp
matrix
