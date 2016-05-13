### cpuECC registration programs

Version compiled with GNU Compiler (gcc) with CMake.

OpenCV binaries included from non standard path: /vol/bitbucket/ic711/usr/local/

Subdirectories:
  * findTransform: Contains cpu ECC program
  * findTransform_gauss: Contains cpu ECC program with gauss optimization 
  * sample_images: contains example 'fruits' images for testing the programs

Process for compiling:
1) enter the appropriate directory (findTransform or findTransform_gauss)
2) run <cmake .> on the command line
3) run <make> on the command line

Generated executable requires commandline inputs in order
1) template image 2) deformed image (will be registered) 3) filename for warp
matrix

Examples:
./findTransform ../sample_images/fruits.jpg ../sample_images/fruits_affine.jpg warpMatrix

./findTransform_gauss ../sample_images/fruits.jpg ../sample_images/fruits_affine.jpg warpMatrix

Corrected image saved to corrected_image.jpg in the working folder.
