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

//#define CORES 16    //replaced by _Ncores below
#define ALIGN8 8
// SCALE: 1, 2, or 4
#define SCALE 2
#define X_PIX 150  //columns -- CHECK if that's enough for Mat, of if need to multiply by channels!
#define Y_PIX 150  //rows


// ============= NEW PARTS (30 March) ================ //



#ifndef __FFT2D_H__
#define __FFT2D_H__

#include <stdint.h>
#include "fft2dlib.h"
#ifndef __HOST__
#include <e_coreid.h>
#endif // __HOST__

#define _Nchips 1                  // # of chips in operand matrix side
#define _Nside  4                  // # of cores in chip side
#define _Ncores (_Nside * _Nside)  // Num of cores = 16

/*
#ifndef _lgSfft
#define _lgSfft 7                  // Log2 of size of 1D-FFT
#endif
#define _Sfft   (1<<_lgSfft)       // Size of 1D-FFT
#define _Score  (_Sfft / _Ncores)  // Num of 1D vectors per-core
#define _Schip  (_Score * _Ncores) // Num of 1D vectors per-chip
#define _Smtx   (_Schip * _Sfft)   // Num of elements in 2D array
*/


#define _Nbanks 2                  // Num of SRAM banks on core

#define _BankA  0
#define _BankW  1
#define _BankB  2
#define _BankP  3
#define _PING   0
#define _PONG   1


//UNLIKELY TO BE CRUCIAL
#ifdef __Debug__
#define dstate(x) { me.mystate = (x); }
#else
#define dstate(x)
#endif

#define TIMERS 10


//UNLIKELY TO BE USEFUL - FFT related
#if 0
#if _lgSfft == 5
#	warning "LgFFt = 5"
#elif _lgSfft == 6
#	warning "LgFFt = 6"
#elif _lgSfft == 7
#	warning "LgFFt = 7"
#endif
#endif






//  CORE
typedef struct {
	int        corenum;
	int        row;
	int        col;

	int        mystate;

	int volatile     go_sync;           // The "go" signal from prev core
        int volatile     sync[_Ncores];     // Sync with peer cores      //DO WE NEED THIS?
	int volatile    *tgt_go_sync;       // ptr to go_sync of next core  //DO WE NEED THIS?
	int volatile    *tgt_sync[_Ncores]; // ptr to sync of target neighbor //DO WE NEED THIS?

	cfloat volatile *bank[_Nbanks][2];            // Ping Pong Bank local space pointer  //DO WE NEED THIS?
	cfloat volatile *tgt_bk[_Ncores][_Nbanks][2]; // Target Bank for matrix rotate in global space //DO WE NEED THIS?

        //uint32_t time_p[TIMERS]; // Timers    //LIKELY NOT NECESSARY
} core_t;


// "ready" and "go" flags for each core
typedef struct {
	volatile int64_t  go;     // Signal to start functions (image processing) in the core
	volatile int      ready;  // Core is ready after reset
} mbox_t;


typedef struct {
	volatile cfloat A[_Smtx]; // Global A matrix
	volatile cfloat B[_Smtx]; // Global B matrix
	volatile mbox_t core;
} shared_buf_t;


typedef struct {
	void            *pBase; // ptr to base of shared buffers
	volatile cfloat *pA;    // ptr to global A matrix
	volatile cfloat *pB;    // ptr to global B matrix
	mbox_t          *pCore; // ptr to cores mailbox
} shared_buf_ptr_t;






// =============== OLD PARTS ========================== //

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
  //Mat whole_image = new Mat(Y_PIX,X_PIX,CV_8UC3); //CV_8UC3 -- 8-bit unsigned 3-channel
  //CHECK if that's enough for Mat, of if need to multiply by channels!
  int whole_image[Y_PIX*X_PIX]; //check what the type should be!!!
  //This might be enough, as we only want to use single-channel images (as
  //it makes no difference if we use rgb or grayscale for our purpose)
} image_strt;


typedef struct
{
  msg_host2dev_t msg_h2d[CORES];
  msg_dev2host_t msg_d2h[CORES];
  text all_text[CORES];
  image_strt core_image[CORES];
} msg_block_t;



