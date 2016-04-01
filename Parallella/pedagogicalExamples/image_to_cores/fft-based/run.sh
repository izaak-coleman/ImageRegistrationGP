#!/bin/bash

set -e

if [[ "$1" == "-d" ]]; then
	Config='Debug'
else
	Config='Release'
fi

#if [[ "$1" == "" ]]; then
#	Img=../../lenna.jpg
#else
#	Img=$@
#fi

if [[ "$1" == "" ]]; then
        mult="1.5"
else
	mult=$@
#       mult="$1"
fi

if [[ "$2" == "" ]]; then
	Img=../../lenna.jpg
else
	Img=$@
fi


ELIBS="${EPIPHANY_HOME}/tools/host/lib"
EHDF="${EPIPHANY_HOME}/bsps/current/platform.hdf"

cd host/${Config}

#sudo -E LD_LIBRARY_PATH=${ELIBS} EPIPHANY_HDF=${EHDF} ./fft2d_host.elf ${Img}
sudo -E LD_LIBRARY_PATH=${ELIBS} EPIPHANY_HDF=${EHDF} ./fft2d_host.elf ${mult} ${Img}

cd ../../

