/*
  epiphany_main.c

  Copyright (C) 2016 Jukka Soikkeli and Chris Smallwood

  [Based on fft2d_main.c by Adapteva]
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


// This program is the accelerator part of the project.
//
// This program runs on the Epiphany system and answers the host with the
// calculation result of the operand matrix.


#include <e-lib.h>
#include "calclib.h"
#include "epiphany.h"
#include "dmalib.h"
#include "dram_buffers.h"
#include "static_buffers.h"

void init();
void calc();

///////////////////////////////
//#define _USE_DRAM_

//#define _USE_DMA_E_
///////////////////////////////

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
	int status;

	status = 0;

	// Initialize data structures - mainly target pointers 
	init();

	do {
		if (me.corenum == 0)
		{
			// Wait for calc() call from the host. When a rising
			// edge is detected in the mailbox, the loop
			// is terminated and a call to the actual
			// calc() function is initiated.
		        while (Mailbox.pCore->go == 0) {};  //busy waiting loop for host to signal "go"

			e_ctimer_set(E_CTIMER_0, E_CTIMER_MAX);
			me.time_p[0] = e_ctimer_start(E_CTIMER_0, E_CTIMER_CLK);


			Mailbox.pCore->ready = 0;

			me.go_sync = (e_group_config.group_rows * e_group_config.group_cols) + 1;
		} else {
			// Wait for "go" from the previous core. When a rising
			// edge is detected in the core's mailbox, the loop
			// is terminated and a call to the actual
			// calc() function is initiated.
			while (me.go_sync == 0) {};
		}
		// Signal "go" to next core.
		*me.tgt_go_sync = me.corenum + 1;

		// Load _Score rows from DRAM.
#ifdef _USE_DRAM_
#	warning "Using DRAM for I/O storage"
#	ifdef _USE_DMA_E_
		dmacpye((void *) &(Mailbox.pA[me.corenum * _Score * _Sedge]), me.bank[_BankA][_PING]);
#	else // _USE_DMA_E_
#		warning "Using rowcpy() instead of DMA_E"
		rowcpy(&(Mailbox.pA[me.corenum * _Score * _Sedge]), me.bank[_BankA][_PING], _Score * _Sedge);
#	endif // _USE_DMA_E_
#endif // _USE_DRAM_

		// ======= OUR CALCULATIONS ========== //

		me.time_p[1] = e_ctimer_get(E_CTIMER_0); //CHECK!!
		calc();  // function for calculations
		me.time_p[2] = e_ctimer_get(E_CTIMER_0); //CHECK!!


		/*
		//LOOP, to test DRAM vs eCore memory speed
		me.time_p[1] = e_ctimer_get(E_CTIMER_0); //CHECK!!
		for(int count=0; count<10; count++) {
		  calc();  // function for calculations
		}
		me.time_p[2] = e_ctimer_get(E_CTIMER_0); //CHECK!!
		*/


		// Save _Score rows to DRAM.
#ifdef _USE_DRAM_
#	ifdef _USE_DMA_E_
		dmacpye(me.bank[_BankA][_PING], (void *) &(Mailbox.pB[me.corenum * _Score * _Sedge]));
#	else // _USE_DMA_E_
#		warning "Using rowcpy() instead of DMA_E"
		rowcpy(me.bank[_BankA][_PING], &(Mailbox.pB[me.corenum * _Score * _Sedge]), _Score * _Sedge);
#	endif // _USE_DMA_E_
#endif // _USE_DRAM_

		// If this is the first core, wait until all cores finished calculation and signal the host.
		if (me.corenum == 0)
		{
			while (me.go_sync == ((e_group_config.group_rows * e_group_config.group_cols) + 1)) {};
			// Signal own End-Of-Calculation to previous core.
			me.go_sync = 0;
	        // Wait until next core ends calculation.
			while (*me.tgt_go_sync > 0) {};

			me.time_p[9] = e_ctimer_stop(E_CTIMER_0);

			Mailbox.pCore->ready = 1;
			Mailbox.pCore->go = 0;
		} else {
	        // If next core ended calculation, signal own End-Of-Calculation to previous core.
			while (*me.tgt_go_sync > 0) {};
			me.go_sync = 0;
		}
	} while (0);

	return status;
}




void calc() {

  int row=0;
	
    
	for(row=0;row<_Score;row++) { 
	  volatile cfloat * restrict xX = (me.bank[_BankA][_PING] + row *_Sedge);

	  for(int col=0;col<_Sedge;col++) { //change the name _Sedge to something else
	    float mult = Mailbox.pCore->mult;

	    int limit  = 255; //limit for image value (255, as 8-bit image elements)
	    int newval = (int) (xX[col]*mult); //value after multiplication
	    if(newval <= limit) //if newval is less than limit, apply
	      xX[col] *= mult;
	    else                //if newval is beyond limit, assign limit value (not enough bits for more)
	      xX[col] = limit;
	  }
	}	

	return;

}


///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
void init()
{
	int row, col, cnum;

	// Initialize the mailbox shared buffer pointers
	Mailbox.pBase = (void *) SHARED_DRAM;
	Mailbox.pA    = Mailbox.pBase + offsetof(shared_buf_t, A[0]);
	Mailbox.pB    = Mailbox.pBase + offsetof(shared_buf_t, B[0]);
	Mailbox.pCore = Mailbox.pBase + offsetof(shared_buf_t, core);

	// Initialize per-core parameters - core data structure
	// Use eLib's e_coreid library's API to retrieve core specific information
	// Use the predefined constants to determine the relative coordinates of the core
	me.row = e_group_config.core_row;
	me.col = e_group_config.core_col;
	me.corenum = me.row * e_group_config.group_rows + me.col;

	// Initialize pointers to the operand matrices ping-pong arrays
	me.bank[_BankA][_PING] = (cfloat *) &(AA[0][0]);
	me.bank[_BankA][_PONG] = (cfloat *) &(BB[0][0]);
	me.bank[_BankW][_PING] = (cfloat *) &(Wn[0]);

	// Use the e_neighbor_id() API to generate the pointer addresses of the arrays
	// in the horizontal and vertical target cores, where the submatrices data will
	// be swapped.
	// CHECK - but likely needed anyway...
	cnum = 0;
	for (row=0; row<e_group_config.group_rows; row++)
		for (col=0; col<e_group_config.group_cols; col++)
		{
			me.tgt_bk[cnum][_BankA][_PING] = e_get_global_address(row, col, (void *) me.bank[_BankA][_PONG]);
			me.tgt_bk[cnum][_BankA][_PONG] = e_get_global_address(row, col, (void *) me.bank[_BankA][_PING]);
			me.tgt_sync[cnum]              = e_get_global_address(row, col, (void *) (&me.sync[me.corenum]));
			cnum++;
		}
	e_neighbor_id(E_NEXT_CORE, E_GROUP_WRAP, &row, &col);
	me.tgt_go_sync = e_get_global_address(row, col, (void *) (&me.go_sync));

	// Generate Wn //CHECK if needed
	if(_lgSedge == 6) {
	generateWn(me.bank[_BankW][_PING], 7);  //TEMP HARDCODED - for testing _lgSedge=6 issues...
	}
	else {
	generateWn(me.bank[_BankW][_PING], _lgSedge);  //CHECK - what do we do with _lgSedge, or its equivalent?
	}


	// Clear the inter-core sync signals
	me.go_sync = 0;
	for (cnum=0; cnum<_Ncores; cnum++)
		me.sync[cnum] = 0;

	// Init the host-accelerator sync signals
	Mailbox.pCore->go = 0;
	Mailbox.pCore->ready = 1;
	//Mailbox.pCore->mult = 1;

#if 0
	// Initialize input image
	for (row=0; row<_Score; row++)
	{
		for (col=0; col<_Sedge; col++)
			*(me.bank[_BankA][_PING] + row * _Sedge + col) = (me.corenum * _Score + row)*1000.0 + col;
		// convert to eDMA
		rowcpy((me.bank[_BankA][_PING] + row * _Sedge), &(Mailbox.pA[(me.corenum * _Score + row) * _Sedge]), _Sedge);
	}
#endif

	return;
}
