/*
Copyright (c) 2016, Jukka Soikkeli
Copyright (c) 2013-2014, Shodruky Rhyammer
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  Neither the name of the copyright holders nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/fb.h>
#include <sys/mman.h>
#include <stdint.h>
#include <e-hal.h>  //eSDK library
#include "data_struct.h" //data structure definitions
#include <opencv2/core/core.hpp> //OpenCV core library, for Mat
//#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#define BUF_OFFSET 0x01000000
#define MAXCORES 16
//#define FBDEV "/dev/fb0"
#define ROWS 4
#define COLS 4
 //#define FRAMES 2000000

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
  e_platform_t eplat;
  e_epiphany_t edev;
  e_mem_t emem; //external memory
  static msg_block_t msg;
  memset(&msg, 0, sizeof(msg));
  struct timespec time;
  double time0, time1;


  //Image reading in part (from OpenCV example at:
  // http://docs.opencv.org/2.4/doc/tutorials/introduction/display_image/display_image.html
  if( argc != 2) {
      cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
      return -1;
    }

  Mat image;
  image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

  if(! image.data ) {
      cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }

  //image.copyTo(image_strt.whole_image); //copies "image" to "image_strt.whole_image"


  // Write image to shared memory
  for(int i=0; i < image.rows; i++) {
    for(int j=0; j < image.cols; j++) {
      //msg.core_image[core];
      //THIS IS NOT RIGHT _ NEED THE FULL MATRIX< SOMEHOW
    }
  }



  // ---- Connect to epiphany ----
  e_init(NULL);  //connect, initilaize HAL data structures
  //e_reset_system(); //(performs full hardware reset of Epiphany) - not sure why...
  e_get_platform_info(&eplat); //gets Epiphany platform info (inserted to "eplat")
  e_alloc(&emem, BUF_OFFSET, sizeof(msg_block_t)); //allocate a buffer in external memory
  
  unsigned int row = 0;
  unsigned int col = 0;

  e_open(&edev, 0, 0, ROWS, COLS); //defines a workgroup "edev" (from 0,0) with ROWSxCOLS cores
  e_write(&emem, 0, 0, 0, &msg, sizeof(msg)); //writes data (of length "sizeof(msg)") to ext mem emem
  e_reset_group(&edev); //Perform soft reset of "edev" workgroup (why?)
  e_load_group("epiphany.srec", &edev, 0, 0, ROWS, COLS, E_TRUE); //RUNS the epiphany.srec program in                                                                  //the workgroup "edev"
  

  // ---- Getting data (the messages) ----
  for (row = 0; row < ROWS; row++) {
    for (col = 0; col < COLS; col++)  {
      unsigned int core = row * COLS + col;
      //e_read(&emem, 0, 0, (off_t)((char *)&msg.all_text[core] - (char *)&msg), &msg.all_text[core], sizeof(text)); //reads data from external memory "emem" [using an offset to get the messages from different cores]
      e_read(&emem, 0, 0, (off_t)((char *)&msg.core_image[core] - (char *)&msg), &msg.core_image[core], sizeof(image_strt)); //reads data from external memory "emem" [using an offset to get the messages from different cores]
      printf(msg.all_text[core].string); //prints the message
    }
  }

  // Getting the "processed" image data out
  Mat out;
  image_strt.whole_image.copyTo(out);




  // ---- Epiphany housekeeping ----
  e_close(&edev); // close the eCore workgroup
  e_free(&emem);  // free the external memory
  e_finalize();   // finalize the connection (disconnect) with epiphany
  return 0;
}
