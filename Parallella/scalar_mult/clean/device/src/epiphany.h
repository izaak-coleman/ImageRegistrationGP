/*
  epiphany.h

  Copyright (C) 2016 Jukka Soikkeli <jes15@ic.ac.uk>

  [Based on fft2d.h by Adapteva]
  Copyright (C) 2012 Adapteva, Inc.
  Contributed by Yainv Sapir <yaniv@adapteva.com>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program, see the file COPYING.  If not, see
  <http://www.gnu.org/licenses/>.
*/


#ifndef __EPIPHANY_H__
#define __EPIPHANY_H__

#include <stdint.h>
//#include "calclib.h" //REMOVE, NOT REQUIRED?
#ifndef __HOST__
#include <e_coreid.h>
#endif // __HOST__

#define _Nchips 1                  // # of chips in operand matrix side //NOT NEEDED?
#define _Nside  4                  // # of cores in chip side
#define _Ncores (_Nside * _Nside)  // Num of cores = 16
#ifndef _lgSedge
#define _lgSedge 7                  // Log2 of size of image edge //WAS 7!, use 6 for 64x64 image, 8 for 256x256
#endif
#define _Sedge   (1<<_lgSedge)      // Size of image edge //i.e. 2^_lgSedge
#define _Score  (_Sedge / _Ncores)  // Num of 1D vectors per-core
#define _Schip  (_Score * _Ncores)  // Num of 1D vectors per-chip
#define _Smtx   (_Schip * _Sedge)   // Num of elements in 2D array

#define _Nbanks 2                  // Num of SRAM banks on core //CHECK - why only 2?

#define _BankA  0
#define _BankW  1
#define _BankB  2
#define _BankP  3
#define _PING   0
#define _PONG   1

#define dstate(x)

#define TIMERS 10

#if 0
#if _lgSedge == 5
#	warning "LgSedge = 5"
#elif _lgSedge == 6
#	warning "LgSedge = 6"
#elif _lgSedge == 7
#	warning "LgSedge = 7"
#endif
#endif

typedef struct {
	int        corenum;
	int        row;
	int        col;

	int        mystate;

	int volatile     go_sync;           // The "go" signal from prev core
	int volatile     sync[_Ncores];     // Sync with peer cores
	int volatile    *tgt_go_sync;       // ptr to go_sync of next core
	int volatile    *tgt_sync[_Ncores]; // ptr to sync of target neighbor

        float volatile *bank[_Nbanks][2];            // Ping Pong Bank local space pointer
  	float volatile *tgt_bk[_Ncores][_Nbanks][2]; // Target Bank for matrix rotate in global space
  //cfloat volatile *bank[_Nbanks][2];            // Ping Pong Bank local space pointer REMOVE complex
  //	cfloat volatile *tgt_bk[_Ncores][_Nbanks][2]; // Target Bank for matrix rotate in global space REMOVE COMPLEX

	uint32_t time_p[TIMERS]; // Timers
} core_t;


typedef struct {
	volatile int64_t  go;     // signal to start calculations
	volatile int      ready;  // Core is ready after reset
  ///////////////////////////////
	volatile float    mult;  // Multiplier
  //////////////////////////////
} mbox_t;


typedef struct {
	volatile float A[_Smtx]; // Global A matrix
	volatile float B[_Smtx]; // Global B matrix
  //volatile cfloat A[_Smtx]; // Global A matrix REMOVE, COMPLEX VER
  //volatile cfloat B[_Smtx]; // Global B matrix REMOVE, COMPLEX VER
	volatile mbox_t core;
} shared_buf_t;


typedef struct {
	void            *pBase; // ptr to base of shared buffers
	volatile float *pA;    // ptr to global A matrix
	volatile float *pB;    // ptr to global B matrix
  //volatile cfloat *pA;    // ptr to global A matrix, REMOVE, COMPLEX VER
  //volatile cfloat *pB;    // ptr to global B matrix, REMOVE, COMPLEX VER
	mbox_t          *pCore; // ptr to cores mailbox
} shared_buf_ptr_t;


#endif // __EPIPHANY_H__
