Image manipulation: Scalar-matrix multiplication [Parallella version]
==================================================

This project provides a simple image manipulation project for the
Parallella, for the purpose of comparing the performance of the
Parallella platform and the Jetson TX1. [comparable code provided for
Jetson TX1]

This uses the Parallella's Epiphany SDK (eSDK) for parallelizing the
work, so a Parallella board is required, with a version of eSDK
(provided with the platform's standard installation package).

## Effect of the program
Input:  an input image, and a multiplier by which the image pixel
        values should be multiplied.

	NOTE: As standard the program runs with images of the size 
	      128x128 pixels as standard. Size 64x64 can be used
	      as well, and was run, but this has to be enabled by
	      changing the value of _lgSedge to 6 on line 36 of
	      "epiphany.h" in the device/src/ subfolder.
	      Size 256x256 does not work, as it does not fit in the
	      local memory of the eCores (at least as implemented).

Result: an output image, which is the original after the intensity 
	change due to the scalar multiplication. Saved in the main
	folder with the name "original_name.out.jpg" [assuming the
	original name was "original_name.jpg"].

Output: Standard output stream includes timing results, for
	1. The computation in the Epiphany chip 
	2. The overhead of putting the image data to the memory [for
	   the Epiphany eCores to access it]
	3. The overhead for taking the result out of the memory
	   [either eCores' local memory, or the shared DRAM].
	4. The overall running time of the program. 


## Requirements:
- Parallella board, tested with eSDK version 
- DevIL library:
  The program uses the DevIL library for reading and writing the image
  file from/to the storage. 
  
  Installing DevIL on Debian-based Linux distributions:
    - type the following commands on a terminal:
      $ sudo apt-get install libdevil-dev
      $ sudo apt-get install libdevil1c2


## Building:
Standard (eCores' local memory version):
- On terminal, run the command: ./build.sh

Shared DRAM version:
- If want to use the shared DRAM instead, there are two macros that
  need to be changed in the main c codes.
  1. host/src/host.c: uncomment line 67, #define _USE_DRAM_
  2. device/src/epiphany_main.c: uncomment line 43, #define _USE_DRAM_

NOTE: These options could be done as inputs to the build.sh script,
but we had to prioritise time use, and left this as it was given this
code is only used for testing purposes.


## Running:
- On terminal, run: ./run.sh <multiplier> <image>
  - the two optional parameters are for the multiplier to be applied
    to the image, and the image to be manipulated
    - multiplier: a float value that specifies a scalar by which
      		  the image matrix should be multiplied. Values < 1
		  lower the pixel intensities, creating a darker
		  image, while > 1 yield a lighter image.
		  Default value: 1.5
    - image:	  image name, in the form "name.jpg".
      		  Default value: "panda.jpg"


## Authors
Jukka Soikkeli and Chris Smallwood, 2016

Based on a project by Yaniv Sapir at Adapteva:
https://github.com/parallella/parallella-examples/tree/master/lena
Copyright 2012, 2014 Adapteva, Inc.


 



