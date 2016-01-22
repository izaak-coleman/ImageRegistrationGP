#########################################################################
# Imperial autoadded bash scripts
#########################################################################
# If not running interactively, don't do anything
[ -z "$PS1" ] && return

# Use NSS shared database for Thunderbird and Firefox
export NSS_DEFAULT_DB_TYPE="sql"

# set your default printer
export PRINTER="ICTMono"

#put your aliases here,
alias pd="pushd"
alias rm="rm -i"

#coloured text
export CLICOLOR=1
export LSCOLORS=ExFxBxDxCxegedabagacad

#########################################################################
# OpenCV compilation PATHs
# Added by Izaak Coleman (ic711), MSc Computing Science
# izaak.coleman11@imperial.ac.uk
# 
# We are required to add the following script to the .bashrc
# of any lab computer we wish to compile/run programs 
# requiring OpenCV 3.1.0.

#extend executable search path
#export PATH=$PATH:/vol/bitbucket/ic711/usr/local/include/opencv2

#extend non dynamic library path
#export LIBRARY_PATH=/vol/bitbucket/ic711/usr/local/lib/

#extend PKG_CONFIG_PATH to include opencv.pc dir
export PKG_CONFIG_PATH=/vol/bitbucket/ic711/usr/local/lib/pkgconfig

#extend ld (library dynamic path) library path
export LD_LIBRARY_PATH=/vol/bitbucket/ic711/usr/local/lib/

##########################################################################
