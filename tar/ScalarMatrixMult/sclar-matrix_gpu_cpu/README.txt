Image manipulation: Scalar-matrix multiplication [cpu and gpu versions]
=================================================

This project provides a simple image manipulation project for cpu and 
nvidia gpus, for the purpose of comparing the performance of the
Parallella platform and the Jetson TX1. [comparable code provided for
Parallella.

## Effect of the program
Input:  an input image, and a multiplier by which the image pixel
        values should be multiplied.

Result: an output image, which is the original after the intensity 
	change due to the scalar multiplication. Saved in the main
	folder with the name "original_name.out.jpg" [assuming the
	original name was "original_name.jpg"].

## Requirements:
- Jetson TX1 or other Cuda-compatable graphics hardware or Intel CPU 
- openCV 3.1.0 installation

## Building:
Standard (eCores' local memory version):
- On terminal, run the command: make

## Running:
- On terminal, run: ./bash_loop_<platform>.sh 
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



