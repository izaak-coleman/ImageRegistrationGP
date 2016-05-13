/* matvecmult_host2.c */
#include <stdio.h>
#include <stdlib.h>
#include <stdcl.h>
inline int parity( int x ) { return ( (x%2)? +1 : -1 ); }
int main()
{
  int i,j;
  unsigned int n = 1024;
  
  /* allocate device-shareable memory */
  float* aa = (float*)clmalloc(stdacc,n*n*sizeof(float),0);
  float* b = (float*)clmalloc(stdacc,n*sizeof(float),0);
  float* c = (float*)clmalloc(stdacc,n*sizeof(float),0);
  
  /* initialize matrix aa[] and vector b[], and zero result vector c[] */
  for(i=0; i<n; i++) for(j=0; j<n; j++) aa[i*n+j] = (1.0/n/n)*i*j*parity(i*j);
  for(i=0; i<n; i++) b[i] = (1.0/n)*i*parity(i);
  for(i=0; i<n; i++) c[i] = 0.0f;
  
  /* sync data with device memory */
  clmsync(stdacc,0,aa,CL_MEM_DEVICE|CL_EVENT_NOWAIT);
  clmsync(stdacc,0,b,CL_MEM_DEVICE|CL_EVENT_NOWAIT);
  clmsync(stdacc,0,c,CL_MEM_DEVICE|CL_EVENT_NOWAIT);
  
  /* perform calculation */
  clndrange_t ndr = clndrange_init1d( 0, 16, 16 );
  clexec(stdacc,0,&ndr,matvecmult_kern2,n,aa,b,c);

  /* sync data with host memory */
  clmsync(stdacc,0,c,CL_MEM_HOST|CL_EVENT_NOWAIT);
 
  /* block until co-processor is done */
  clwait(stdacc,0,CL_ALL_EVENT);
  for(i=0;i<n;i++) printf("%d %f %f\n",i,b[i],c[i]);
  clfree(aa);
  clfree(b);
  clfree(c);
}
