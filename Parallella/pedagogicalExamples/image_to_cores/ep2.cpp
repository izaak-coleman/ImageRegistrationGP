/*
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


#include "e_lib.h"
#include "data_struct.h"
#include "string.h"

#define BUF_ADDRESS 0x8f000000
#define MAXI 64
#define ROWS 4
#define COLS 4


int main(void)
{
  e_coreid_t coreid;
  coreid = e_get_coreid();
  unsigned int row, col, core, cores;
  e_coords_from_coreid(coreid, &row, &col);
  core = row * COLS + col;
  cores = ROWS * COLS;

  //volatile msg_block_t *msg = (msg_block_t *)BUF_ADDRESS;
  msg_block_t *msg = (msg_block_t *)BUF_ADDRESS;

  //msg->msg_d2h[core].coreid = coreid;
  //msg->msg_d2h[core].value = frame;

  char str[50] = {'H','e','l','l','o',' ','f','r','o','m',' ','e','C','o','r','e',' ',core+48,'!','\n','\0'};

  strcpy(msg->all_text[core].string,str);


  return 0;
}
