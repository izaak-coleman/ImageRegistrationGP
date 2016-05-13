//COPRTHR example: Programming Parallella using STDCL
/* matvecmult_kern2.cl */

#include <stdcl.h>
void matvecmult_kern2( unsigned int n, float* aa, float* b, float* c )
{
  int i,j,k;
  k = get_global_id(0);

    int n16 = n/16;
    int m16 = n%16;
    int ifirst = k*n16 + ((k>m16)? 0:k);
    int iend = ifirst + n16 + ((k<m16)? 1:0);
    for(i=ifirst; i<iend; i++) {
      float tmp = 0.0f;
      for(j=0; j<n; j++) tmp += aa[i*n+j] * b[j];
      c[i] = tmp;
    }
}
