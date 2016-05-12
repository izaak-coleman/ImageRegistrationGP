/*
  static_buffers.c

  Copyright (c) 2016 Jukka Soikkeli <jes15@ic.ac.uk>
  
  Based on original by Adapteva:
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


#include "static_buffers.h"


// Here's an example of explicit placement of static objects in memory. The three matrices
// are placed in the respective SRAM banks. However, if necessary, the linker may relocate
// the objects wherever within the bank. The core structure "me" is specifically located
// at an explicit address - 0x7000. To do that, a custom linker file (LDF) was defined,
// based on a standard LDF, in which a special data section was added with the required
// address to assign to the "me" object.
volatile float  AA[_Score][_Sedge] ALIGN(8) SECTION(".data_bank1");       // local operand array (ping)
volatile float  BB[_Score][_Sedge] ALIGN(8) SECTION(".data_bank2");       // local operand array (pong)

//REMOVED COMPLEX VERSIONS:
core_t           me                ALIGN(8) SECTION("core_data_section"); // core data structure
shared_buf_ptr_t Mailbox           ALIGN(8) SECTION("core_data_section"); // Mailbox pointers
volatile e_dma_desc_t tcb          ALIGN(8) SECTION("core_data_section"); // TCB structure for DMA
