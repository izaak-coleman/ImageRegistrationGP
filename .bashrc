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

#extend executable search path
#export PATH=$PATH:/vol/bitbucket/ic711/usr/local/include/opencv2

#extend non dynamic library path
#export LIBRARY_PATH=/vol/bitbucket/ic711/usr/local/lib/

#extend PKG_CONFIG_PATH to include opencv.pc dir
export PKG_CONFIG_PATH=/vol/bitbucket/ic711/usr/local/lib/pkgconfig

#extend ld (library dynamic path) library path to include the opencv locatoin
export LD_LIBRARY_PATH=/vol/bitbucket/ic711/usr/local/lib/:/vol/cuda/6.5.14/lib64:/vol/cuda/6.5.14/lib


#path to bash bin dir
export PATH=$PATH:/vol/cuda/6.5.14/bin



# run the set up script 
if [ -f /vol/cuda/6.5.14/setup.sh ]
  then
	. /vol/cuda/6.5.14/setup.sh
fi

