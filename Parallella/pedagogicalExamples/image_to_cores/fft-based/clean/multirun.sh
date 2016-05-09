#!/bin/bash

alg_time=0.0
alg_time_t=0.0
TIMEFORMAT=%R  # Time format of unix time used: %R "real time" (not CPU time)

# Variables to be changed as needed
process="./run.sh 1.5"  # the process to be timed
N=15               # number of runs the time is averaged over

# Executing process "./sorting" in a loop, to calculate average execution time
for ((i=1; i<=N; i++))
do
   #echo "Loop iteration $i"
   #alg_time=$(time ($process >/dev/null 2>&1) 2>&1) #time the process, with time
   alg_time=$(time ($process >>out.txt 2>&1) 2>&1) #time the process, with time
   alg_time_t=$(echo $alg_time_t+$alg_time | bc)    #add to total time, using bc
done

# Echo total time
echo "Total time of $N runs of process $process is $alg_time_t"

# Calculate average time, and echo it
avg_time=$(echo "scale=5; $alg_time_t/$N" | bc) #calculate average time using bc
echo "Average time is $avg_time"



# NOTES: ----------------------------------------------------------------------
# "scale=5" sets scale to be 5 digits after decimal point in bc
# ">/dev/null" redirects the output streams of the process to the "null device"
# "2>&1" redirects stderr to stdout
# More on I/O redirection: http://www.tldp.org/LDP/abs/html/io-redirection.html

# Power consumption: powerstat
# http://www.hecticgeek.com/2012/02/powerstat-power-calculator-ubuntu-linux/

# FOR nVidia gaphics cards: nvidia-smi
# shows statistics on memory usage, GPU utilization, temperature
# http://unix.stackexchange.com/questions/38560/gpu-usage-monitoring-cuda
