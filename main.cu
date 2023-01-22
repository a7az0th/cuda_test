#include <stdio.h>

__global__ void vector_add(float *a, float *out, int n) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i >= n) {
        return;
    }
    out[i] = a[i] + a[i];
}

#define checkErr(X) \
{ \
  cudaError_t err = X;\
  if (err != cudaSuccess) { \
    printf("%s[%d] CUDA Error %d\n", __func__, __LINE__, err); \
    exit(err); \
  } \
}

int main() {
    const size_t N = 10;	
    float a[N];
    float *d_a = nullptr;
    float *d_b = nullptr;

    for (int i = 0; i < N; i++) {
        a[i] = i;
    }

    // Allocate device memory for a
    checkErr(cudaMalloc((void**)&d_a, sizeof(float) * N));
    checkErr(cudaMalloc((void**)&d_b, sizeof(float) * N));

    // Transfer data from host to device memory
    checkErr(cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice));

    vector_add<<<1, N>>>(d_a, d_b, N);

    checkErr(cudaMemcpy(a, d_b, sizeof(float) * N, cudaMemcpyDeviceToHost));

    // Cleanup after kernel execution
    checkErr(cudaFree(d_a));
    checkErr(cudaFree(d_b));

    checkErr(cudaDeviceSynchronize());

    for (int i = 0; i < N; i++) {
        printf("%.2f, ", a[i]);
    }
    printf("\n");
    return 0;
}
