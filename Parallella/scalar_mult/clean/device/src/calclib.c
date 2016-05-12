/*
  calclib.c

  Copyright (c) 2016 Jukka Soikkeli <jes15@ic.ac.uk>

  [Based on ff2dlib.c by Adapteva]
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


#include "calclib.h"
#include "static_buffers.h"

// Generate Wn array
/*
void generateWn(volatile cfloat * restrict Wn, int lgNN)
{
	int   Wc, NN;
	float C, S;

	NN = 1 << lgNN;

	C = Cw_lg[lgNN]; //  Cos(2*Pi/(2^lgNN))
	S = Sw_lg[lgNN]; // -Sin(2*Pi/(2^lgNN))

	Wn[0] = 1.0 + 0.0 * I;
	Wn[0 + (NN >> 1)] = conj(Wn[0]);
	for (Wc=1; Wc<(NN >> 1); Wc++) {
		Wn[Wc] = (C * crealf(Wn[Wc-1]) - S * cimagf(Wn[Wc-1])) +
		         (S * crealf(Wn[Wc-1]) + C * cimagf(Wn[Wc-1])) * I;
		Wn[Wc + (NN >> 1)] = conj(Wn[Wc]);
	}

	return;
}


#if !((defined _USE_DMA_I_) && (defined _USE_DMA_E_))
#	warning "Using rowcopy() instead of DMA"
// Copy a row of length NN
void rowcpy(volatile cfloat * restrict a, volatile cfloat * restrict b, int NN)
{
	int i;

	for (i=0; i<NN; i++)
		b[i] = a[i];

	return;
}
#endif


#ifdef __HOST__
// Subtract two NNxNN matrices c = a - b
void matsub(volatile float * restrict a, volatile float * restrict b, volatile float * restrict c, int NN)
{
	int i, j;

	for (i=0; i<NN; i++)
		for (j=0; j<NN; j++)
			c[i*NN+j] = a[i*NN+j] - b[i*NN+j];
	
	return;
}


// Check if a NNxNN matrix is zero
int iszero(volatile float * restrict a, int NN)
{
	int i, j, z;

	z = 0;
	for (i=0; i<NN; i++)
		for (j=0; j<NN; j++)
			if (fabs(a[i*NN+j]) > EPS)
				z = z | 1;

	return (!z);
}
#endif // __HOST__
*/
