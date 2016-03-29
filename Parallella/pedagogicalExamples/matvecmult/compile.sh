clcc -k -c matvecmult_kern2.cl
gcc -o matvecmult2.x matvecmult_host2.c matvecmult_kern2.o -I$COPRTHR_PATH/include -L$COPRTHR_PATH/lib -lstdcl -locl
