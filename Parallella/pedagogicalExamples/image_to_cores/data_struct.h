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

#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <stdint.h>

#define CORES 16
#define ALIGN8 8
// SCALE: 1, 2, or 4
#define SCALE 2
#define X_PIX 150  //columns -- CHECK if that's enough for Mat, of if need to multiply by channels!
#define Y_PIX 150  //rows

typedef struct __attribute__((aligned(ALIGN8)))
{
  uint32_t value;
  uint32_t coreid;
} msg_dev2host_t;

typedef struct __attribute__((aligned(ALIGN8)))
{
  uint32_t value;
} msg_host2dev_t;

// TESTING STRUCT - this is used for the Hello messages
typedef struct __attribute__((aligned(ALIGN8))) {
  char string[50];
} text;


// TESTING STRUCT - this is used for the Mat object
typedef struct __attribute__((aligned(ALIGN8))) {
  Mat whole_image = new Mat(Y_PIX,X_PIX,CV_8UC3); //CV_8UC3 -- 8-bit unsigned 3-channel
  //CHECK if that's enough for Mat, of if need to multiply by channels!
} image_strt;


typedef struct
{
  msg_host2dev_t msg_h2d[CORES];
  msg_dev2host_t msg_d2h[CORES];
  text all_text[CORES];
  image_strt core_image[CORES];
} msg_block_t;



