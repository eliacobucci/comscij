// nvcc -O3 -arch=sm_70 -o vecadd cuda_vecadd.cu
#include <cstdio>
__global__ void vecadd(const float* a, const float* b, float* c, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) c[i] = a[i] + b[i];
}
int main(){
    const int n = 1<<26; // ~67M
    size_t bytes = n*sizeof(float);
    float *a,*b,*c;
    cudaMalloc(&a, bytes);
    cudaMalloc(&b, bytes);
    cudaMalloc(&c, bytes);
    vecadd<<<(n+255)/256, 256>>>(a,b,c,n);
    cudaDeviceSynchronize();
    printf("OK\n");
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
