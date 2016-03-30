clcc -k -c matvecmult_kern.cl
gcc -o matvecmult.x matvecmult_host.c matvecmult_kern.o -I$COPRTHR_PATH/include -L$COPRTHR_PATH/lib -lstdcl -locl
