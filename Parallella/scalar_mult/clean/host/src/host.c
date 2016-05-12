/*
  host.c

  Copyright (c) 2016 Jukka Soikkeli <jes15@ic.ac.uk>

  [Based on fft2d_host.c by Adapteva]
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


// This is the host part of the Parallella image manipulation project.
//
// This program runs on the linux host and invokes the Epiphany calc()
// implementation. It communicates with the system via the eHost library.

// **********************************
// To install DevIL on Linux, type:
//
// sudo apt-get install libdevil-dev
// sudo apt-get install libdevil1c2
// **********************************

#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <sys/time.h>
#include <time.h>

#include <IL/il.h>
#define ILU_ENABLED
#define ILUT_ENABLED
/* We would need ILU just because of iluErrorString() function... */
/* So make it possible for both with and without ILU!  */
#ifdef ILU_ENABLED
#include <IL/ilu.h>
#define PRINT_ERROR_MACRO printf("IL Error: %s\n", iluErrorString(Error))
#else /* not ILU_ENABLED */
#define PRINT_ERROR_MACRO printf("IL Error: 0x%X\n", (unsigned int) Error)
#endif /* not ILU_ENABLED */
#ifdef ILUT_ENABLED
#include <IL/ilut.h>
#endif

/////////////////////////////
// FOR USING DRAM instead of local mem
//#define _USE_DRAM_
/////////////////////////////

typedef unsigned int e_coreid_t;
#include <e-hal.h>
#include "calclib.h"
#include "epiphany.h"
#include "dram_buffers.h"

#define FALSE 0
#define TRUE  1

typedef struct {
	int  run_target;
	e_loader_diag_t verbose;
	char srecFile[4096];
	char ifname[255];
	char ofname[255];
} args_t;

args_t ar = {TRUE, L_D0, ""};
void get_args(int argc, char *argv[]);


#define eMHz 600e6
#define RdBlkSz 1024

int  main(int argc, char *argv[]);
void matrix_init(int seed);
int  calc_go(e_mem_t *pDRAM); //function to signal calculations to start
void init_coreID(e_epiphany_t *pEpiphany, unsigned int *coreID, int rows, int cols, unsigned int base_core);

//cfloat Bref[_Smtx];
//cfloat Bdiff[_Smtx];

unsigned int DRAM_BASE  = 0x8f000000;
unsigned int BankA_addr = 0x2000;  //address of memory bank A
unsigned int coreID[_Ncores];      //core ID array

typedef struct timeval timeval_t;
e_platform_t platform;

//////////////////////////
	float mult = 1.5; //multiplier
//////////////////////////

int main(int argc, char *argv[])
{
	e_epiphany_t Epiphany, *pEpiphany;
	e_mem_t      DRAM,     *pDRAM;
	unsigned int msize;
	int          row, col, cnum;


	//DevIL library related...
	ILuint  ImgId;
	ILubyte *imdata;
	ILuint  imsize, imBpp;


	unsigned int addr;
	size_t sz;
	timeval_t timer[8];
	uint32_t time_d[TIMERS];
	FILE *fo;
	int  result;
	//////////////////////////////////////////
	gettimeofday(&timer[0], NULL);
	//////////////////////////////////////////
	pEpiphany = &Epiphany;
	pDRAM     = &DRAM;
	msize     = 0x00400000;

	get_args(argc, argv);

	fo = fopen("matprt.m", "w");
	if ((fo == NULL))
	{
		fprintf(stderr, "Could not open Octave file \"%s\" ...exiting.\n", "matprt.m");
		exit(4);
	}


	// Connect to device for communicating with the Epiphany system
	// Prepare device
	e_set_loader_verbosity(ar.verbose);
	e_init(NULL);
	e_reset_system();
	e_get_platform_info(&platform);
	if (e_open(pEpiphany, 0, 0, platform.rows, platform.cols))
	{
		fprintf(fo, "\nERROR: Can't establish connection to Epiphany device!\n\n");
		exit(1);
	}
	if (e_alloc(pDRAM, 0x00000000, msize))
	{
		fprintf(fo, "\nERROR: Can't allocate Epiphany DRAM!\n\n");
		exit(1);
	}

	// Initialize Epiphany "Ready" state
	addr = offsetof(shared_buf_t, core.ready);
	Mailbox.core.ready = 0;
	e_write(pDRAM, 0, 0, addr, (void *) &(Mailbox.core.ready), sizeof(Mailbox.core.ready));

	////////////////////////////////////////
	// Pass the multiplier into the program
	printf("Multiplier is %1.1f \n",mult);
	addr = offsetof(shared_buf_t, core.mult);
	Mailbox.core.mult = mult;
	e_write(pDRAM, 0, 0, addr, (void *) &(Mailbox.core.mult), sizeof(Mailbox.core.mult));
	////////////////////////////////////////


	// Load program
	printf("Loading program on Epiphany chip...\n");
	strcpy(ar.srecFile, "../../device/Release/e_calc.srec");
	result = e_load_group(ar.srecFile, pEpiphany, 0, 0, platform.rows, platform.cols, (e_bool_t) (ar.run_target));
	if (result == E_ERR) {
		printf("Error loading Epiphany program.\n");
		exit(1);
	}


	// Check if the DevIL shared lib's version matches the executable's version.
	if (ilGetInteger(IL_VERSION_NUM) < IL_VERSION)
	{
		fprintf(stderr, "DevIL version is different ...exiting!\n");
		exit(2);
	}

	// Initialize DevIL.
	ilInit();
#ifdef ILU_ENABLED
	iluInit();
#endif



	// create the coreID list
	init_coreID(pEpiphany, coreID, _Nside, _Nside, 0x808);


	// Generate the main image name to use, bind it and load the image file.
	ilGenImages(1, &ImgId);
	ilBindImage(ImgId);
	printf("\n");
	printf("Loading original image from file \"%s\".\n\n", ar.ifname);
	if (!ilLoadImage(ar.ifname))
	{
		fprintf(stderr, "Could not open input image file \"%s\" ...exiting.\n", ar.ifname);
		exit(3);
	}


	// Display the image's dimensions to the end user.
	printf("Width: %d  Height: %d  Depth: %d  Bpp: %d\n\n",
	       ilGetInteger(IL_IMAGE_WIDTH),
	       ilGetInteger(IL_IMAGE_HEIGHT),
	       ilGetInteger(IL_IMAGE_DEPTH),
	       ilGetInteger(IL_IMAGE_BITS_PER_PIXEL));

	imdata = ilGetData();
	imsize = ilGetInteger(IL_IMAGE_WIDTH) * ilGetInteger(IL_IMAGE_HEIGHT);
	imBpp  = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);

	if (imsize != (_Sedge * _Sedge))
	{
		printf("Image file size is different from %dx%d ...exiting.\n", _Sedge, _Sedge);
		exit(5);
	}


	// Extract image data into the A matrix.
	for (unsigned int i=0; i<imsize; i++)
	{
	  //Mailbox.A[i] = (float) imdata[i*imBpp] + 0.0 * I; //REMOVE, imaginary version
	  Mailbox.A[i] = (float) imdata[i*imBpp];
	}

	fprintf(fo, "\n");


	// Generate operand matrices based on a provided seed
	matrix_init(0);

#ifdef _USE_DRAM_
	// Copy operand matrices to Epiphany system
        time_t upload_overhead; //ADDED
	upload_overhead = clock(); //ADDED
	gettimeofday(&timer[1], NULL);
	addr = offsetof(shared_buf_t, A[0]);
	sz = sizeof(Mailbox.A);
	printf(        "DRAM in use\n"); //REMOVE
	 printf(       "Writing A[%ldB] to address %08x...\n", sz, addr);
	fprintf(fo, "%% Writing A[%ldB] to address %08x...\n", sz, addr);
	e_write(pDRAM,0,0,addr, (void *) Mailbox.A, sz); //added "pDRAM,0,0"

	addr = offsetof(shared_buf_t, B[0]);
	sz = sizeof(Mailbox.B);
	 printf(       "Writing B[%ldB] to address %08x...\n", sz, addr);
	fprintf(fo, "%% Writing B[%ldB] to address %08x...\n", sz, addr);
	e_write(pDRAM,0,0,addr, (void *) Mailbox.B, sz); //added "pDRAM,0,0"
	gettimeofday(&timer[2], NULL);
        printf( "\nUpload time taken = %f sec\n", (clock() - upload_overhead)/(double)CLOCKS_PER_SEC); //ADDED
#else
	// Copy operand matrices to Epiphany cores' memory
	 printf(       "Writing image to Epiphany\n");
	fprintf(fo, "%% Writing image to Epiphany\n");
        time_t upload_overhead; //ADDED        
	upload_overhead = clock(); //ADDED
	gettimeofday(&timer[1], NULL);
	sz = sizeof(Mailbox.A) / _Ncores;
	for (row=0; row<(int) platform.rows; row++)
		for (col=0; col<(int) platform.cols; col++)
		{
			addr = BankA_addr;
			printf(".");
			fflush(stdout);
			cnum = e_get_num_from_coords(pEpiphany, row, col);
			fprintf(fo, "%% Writing A[%uB] to address %08x...\n", sz, (coreID[cnum] << 20) | addr); fflush(fo);
			e_write(pEpiphany, row, col, addr, (void *) &Mailbox.A[cnum * _Score * _Sedge], sz);
		}
	gettimeofday(&timer[2], NULL);
        printf( "\nUpload time taken = %f sec\n", (clock() - upload_overhead)/(double)CLOCKS_PER_SEC); //ADDED
	printf("\n");
#endif


	// Call the Epiphany calc() function
	 printf(       "GO!\n");
	fprintf(fo, "%% GO!\n");
	fflush(stdout);
	fflush(fo);
        time_t function; //ADDED
	function = clock(); //ADDED
	gettimeofday(&timer[3], NULL);
        calc_go(pDRAM);  //checks the DRAM for ready and go, runs the core programs, and waits for all cores to finish
	gettimeofday(&timer[4], NULL); //timing the whole Epiphany calculation using sys/time
        printf( "Function time taken = %f sec\n", (clock() - function)/(double)CLOCKS_PER_SEC); //ADDED
	 printf(       "Done!\n\n");
	fprintf(fo, "%% Done!\n\n");
	fflush(stdout);
	fflush(fo);


	//TIMERS on Epiphany

	// *** "Original type" of timers - seem incorrect and give weird results
	/*
	// Read time counters
	fprintf(fo, "%% Reading time count...\n");
	addr = 0x7128+0x4*2 + offsetof(core_t, time_p[0]);
	sz = TIMERS * sizeof(uint32_t);
	e_read(pEpiphany, 0, 0, addr, (void *) (&time_p[0]), sz);


	//time_d[7] = time_p[1] - time_p[2]; // calc()
	//time_d[9] = time_p[0] - time_p[9]; // Total cycles

	//printf(       "Finished calculation in %u cycles (%5.3f msec @ %3.0f MHz)\n\n", time_d[9], (time_d[9] * 1000.0 / eMHz), (eMHz / 1e6));
	//fprintf(fo, "%% Finished calculation in %u cycles (%5.3f msec @ %3.0f MHz)\n\n", time_d[9], (time_d[9] * 1000.0 / eMHz), (eMHz / 1e6));
	//printf(       "Calculations     - %7u cycles (%5.3f msec)\n", time_d[7], (time_d[7] * 1000.0 / eMHz));
	*/
	// ***

	// Timers using sys/time.h [sensible results, robust]
	 printf(       "Memory overhead (in)  - (%5.3f msec)\n", (timer[2].tv_usec - timer[1].tv_usec)/1000.0);
	 printf(       "Processing time on Epiphany     - (%5.3f msec)\n", (timer[4].tv_usec - timer[3].tv_usec)/1000.0);
	 printf(       "\n");

	 printf(       "Reading processed image back to host\n");
	fprintf(fo, "%% Reading processed image back to host\n");



	// Read result matrix
#ifdef _USE_DRAM_
	gettimeofday(&timer[5], NULL);
        time_t download_overhead; //ADDED
        download_overhead = clock(); //ADDED
	addr = offsetof(shared_buf_t, B[0]);
	sz = sizeof(Mailbox.B);
	 printf(       "Reading B[%ldB] from address %08x...\n", sz, addr);
	fprintf(fo, "%% Reading B[%ldB] from address %08x...\n", sz, addr);
	int blknum = sz / RdBlkSz;
	int remndr = sz % RdBlkSz;
	int i;
	for (i=0; i<blknum; i++)
	{
		printf(".");
		fflush(stdout);
		e_read(pDRAM,0,0,addr+i*RdBlkSz, (void *) ((long unsigned)(Mailbox.B)+i*RdBlkSz), RdBlkSz);
	}
	printf(".");
	fflush(stdout);
	e_read(pDRAM,0,0,addr+i*RdBlkSz, (void *) ((long unsigned)(Mailbox.B)+i*RdBlkSz), remndr);
	gettimeofday(&timer[6], NULL);
        printf( "\nDownload overhead time taken = %f sec\n", (clock() - download_overhead)/(double)CLOCKS_PER_SEC); //ADDED

#else
	// Read result matrix from Epiphany cores' memory
	gettimeofday(&timer[5], NULL);
        time_t download_overhead; //ADDED
        download_overhead = clock(); //ADDED
	sz = sizeof(Mailbox.A) / _Ncores;
	for (row=0; row<(int) platform.rows; row++)
		for (col=0; col<(int) platform.cols; col++)
		{
			addr = BankA_addr;
			printf(".");
			fflush(stdout);
			cnum = e_get_num_from_coords(pEpiphany, row, col);
			fprintf(fo, "%% Reading A[%uB] from address %08x...\n", sz, (coreID[cnum] << 20) | addr); fflush(fo);
			e_read(pEpiphany, row, col, addr, (void *) &Mailbox.B[cnum * _Score * _Sedge], sz);
		}
	gettimeofday(&timer[6], NULL);
        printf( "\nDownload overhead time taken = %f sec\n", (clock() - download_overhead)/(double)CLOCKS_PER_SEC); //ADDED
#endif
	printf("\n");
	 printf(       "Memory overhead (out)  - (%5.3f msec)\n", (timer[6].tv_usec - timer[5].tv_usec)/1000.0);


	// Convert processed image matrix B into the image file date.
	for (unsigned int i=0; i<imsize; i++)
	{
		for (unsigned int j=0; j<imBpp; j++)
			imdata[i*imBpp+j] = cabs(Mailbox.B[i]);
	}

	// Save processed image to the output file.
	ilEnable(IL_FILE_OVERWRITE);
	printf("\nSaving processed image to file \"%s\".\n\n", ar.ofname);
	if (!ilSaveImage(ar.ofname))
	{
		fprintf(stderr, "Could not open output image file \"%s\" ...exiting.\n", ar.ofname);
		exit(7);
	}

	// We're done with the image, so let's delete it.
	ilDeleteImages(1, &ImgId);


	// Close connection to device
	if (e_close(pEpiphany))
	{
		fprintf(fo, "\nERROR: Can't close connection to Epiphany device!\n\n");
		exit(1);
	}
	if (e_free(pDRAM))
	{
		fprintf(fo, "\nERROR: Can't release Epiphany DRAM!\n\n");
		exit(1);
	}

	fflush(fo);
	fclose(fo);

	///////////////////////////////////////
	gettimeofday(&timer[7], NULL);
	printf(       "Full program time     - (%5.3f msec)\n", (timer[7].tv_usec - timer[0].tv_usec)/1000.0 );
	///////////////////////////////////////
	
	//Returning success if test runs expected number of clock cycles
	if(time_d[9]>50000){
	  return EXIT_SUCCESS;
	}
	else{
	  return EXIT_FAILURE;
	}
}
	
// Call (invoke) the calc() function
int calc_go(e_mem_t *pDRAM)
{
	unsigned int addr;
	size_t sz;

	printf("Waiting for Epiphany to be ready.\n");
	// Wait until Epiphany is ready
	addr = offsetof(shared_buf_t, core.ready);
	Mailbox.core.ready = 0;
	while (Mailbox.core.ready == 0)
		e_read(pDRAM, 0, 0, addr, (void *) &(Mailbox.core.ready), sizeof(Mailbox.core.ready));
	//READS THE READY MESSAGE UNTIL READY - busy waiting, but OK as the cores are only doing one thing

	printf("Waiting until cores finished calc.\n");
	// Wait until cores finished previous calculation
	addr = offsetof(shared_buf_t, core.go);
	sz = sizeof(int64_t);
	Mailbox.core.go = 1;
	while (Mailbox.core.go != 0)
		e_read(pDRAM, 0, 0, addr, (void *) (&Mailbox.core.go), sz);

	printf("Signal cores to start again.\n");	
	// Signal cores to start crunching   //same as above, but does a WRITE instead or read
	Mailbox.core.go = 1;
	addr = offsetof(shared_buf_t, core.go);
	sz = sizeof(int64_t);
	e_write(pDRAM, 0, 0, addr, (void *) (&Mailbox.core.go), sz);
	

	// Wait until cores finished calculation
	addr = offsetof(shared_buf_t, core.go); //REPETITION?
	sz = sizeof(int64_t);  //REPETITION?
	Mailbox.core.go = 1;  //REPETITION?
	while (Mailbox.core.go != 0)
		e_read(pDRAM, 0, 0, addr, (void *) (&Mailbox.core.go), sz);

	return 0;
}


// Initialize result matrices
void matrix_init(int seed)
{
	int i, j, p;

	p = 0;
	for (i=0; i<_Sedge; i++)
		for (j=0; j<_Sedge; j++)
			Mailbox.B[p++] = 0x8dead;

	return;
}


// Generate Epiphany Core ID's
void init_coreID(e_epiphany_t *pEpiphany, unsigned int *coreID, int rows, int cols, unsigned int base_core)
{
	unsigned int cnum;
	unsigned int row, col;
	unsigned int base_row, base_col;

	base_row = base_core >> 6;
	base_col = base_core & 0x3f;

	cnum = 0;
	for (row=base_row; row<base_row+rows; row++)
		for (col=base_col; col<base_col+cols; col++)
		{
			cnum = e_get_num_from_coords(pEpiphany, row, col);
			coreID[cnum] = (row << 6) | (col << 0);
		}

		return;
}


// Print a usage message
void usage(int n)
{
        printf("Usage: host.elf [-no-run] [-verboseN] [-h] <multiplier> <image-file>\n");
	printf("   -verbose0, -verbose1, -verbose2, -verbose3: available levels of diagnostics\n");
	printf("\n");
	printf("   The program will open the image file ""image-file"", multiply pixel values\n");
	printf("   by the multiplier ""multiplier"" and save the resulting image to ""image-file.out.jpg""\n\n");
	exit(n);
}


// Process command line args
void get_args(int argc, char *argv[])
{
	int n;
	char buf[255];
	char *dotp;

	////////////////////////
	//If more than two args, then multiplier is one of them
	if(argc>2)
	  mult = atof(argv[1]);
	////////////////////////

	for (n=2; n<argc; n++)
	{
		if (!strcmp(argv[n], "-no-run"))
			ar.run_target = FALSE;

		if (!strcmp(argv[n], "-verbose0"))
			ar.verbose = L_D0;

		if (!strcmp(argv[n], "-verbose1"))
			ar.verbose = L_D1;

		if (!strcmp(argv[n], "-verbose2"))
			ar.verbose = L_D2;

		if (!strcmp(argv[n], "-verbose3"))
			ar.verbose = L_D3;

		if (!strcmp(argv[n], "-h"))
			usage(0);

		strcpy(ar.ifname, argv[n]);
	}


	// Process input arguments
	if (!strcmp(ar.ifname, ""))
	{
		usage(1);
	} else {
		strcpy(ar.ofname, ar.ifname);
		dotp = strrchr(ar.ofname, '.');
		if (dotp != NULL)
		{
			strcpy(buf, dotp);
			strcpy(dotp, ".out");
			strcat(ar.ofname, buf);
		} else {
			strcat(ar.ofname, ".out");
		}
	}


	return;
}
