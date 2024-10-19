#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "common.h"

#define TILE_WIDTH 24

__global__ void matmul_forward_tiled_kernel(float* out, float* A, float* B, 
                               int I, int J, int K){
    // out = A * B.T
    // A is of size (I,J)
    // B is of size (K,J)
    // out will be of size (I,K)
    // simple matmul forward kernel

    __shared__ float lhs_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ float rhs_s[TILE_WIDTH][TILE_WIDTH];

    int A_row = TILE_WIDTH * blockIdx.y + threadIdx.y;
    int B_col = TILE_WIDTH * blockIdx.x + threadIdx.x;


    float val = 0.0f;
    for (int m = 0; m < CEIL_DIV(J, TILE_WIDTH); m++){
        if (A_row < I && m * TILE_WIDTH + threadIdx.x < J)
            lhs_s[threadIdx.y][threadIdx.x] = A[A_row * J + m * TILE_WIDTH + threadIdx.x];
        else
            lhs_s[threadIdx.y][threadIdx.x] = 0.0f;

        if (B_col < K && m * TILE_WIDTH + threadIdx.y < J)
            rhs_s[threadIdx.x][threadIdx.y] = B[B_col * J + m * TILE_WIDTH + threadIdx.y];
        else
            rhs_s[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++){
            val += lhs_s[threadIdx.y][i] * rhs_s[threadIdx.x][i];
        }
        __syncthreads();
    }
    if (A_row < I && B_col < K)
        out[A_row * K + B_col] = val;
}
        


void matmul_forward_tiled(float* out, float* A, float* B, 
                    int I, int J, int K, int sqrt_block_size){
    
    dim3 gridDim(CEIL_DIV(K, sqrt_block_size), CEIL_DIV(I, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_tiled_kernel<<<gridDim, blockDim>>>(out, A, B, I, J, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
    }

int main(){
    srand(0);

    int I = 64;
    int J = 256;
    int K = 512;

    int deviceIdx = 0;
    CHECK_CUDA_ERROR(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // create host memory of random numbers
    float* out = (float*)malloc(I * K * sizeof(float));
    float* A = make_random_float(I * J);
    float* B = make_random_float(K * J);

    // move to GPU
    float* d_out;
    float* d_A;
    float* d_B;

    CHECK_CUDA_ERROR(cudaMalloc(&d_out, I * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, I * J * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * J * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, I * J * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, B, K * J * sizeof(float), cudaMemcpyHostToDevice));


    // // first check the correctness of the kernel
    matmul_forward_cpu(out, A, B, I, J, K);

    printf("Checking block size %d x %d.\n", TILE_WIDTH, TILE_WIDTH);
    matmul_forward_tiled(d_out, d_A, d_B, I, J, K, TILE_WIDTH);
    validate_result(d_out, out, I * K, 1e-4);

    int repeat_times = 100;
    float elapsed_time = benchmark_kernel(repeat_times, matmul_forward_tiled,
                                            d_out, d_A, d_B, I, J, K, TILE_WIDTH);
    
    float tflops = (float)I * (2*J - 1) * K  / elapsed_time * 1e3f / 1e12f;
    printf("sqrt_block_size %4d | time %.4f ms | tflops %.2f \n", TILE_WIDTH, elapsed_time, tflops);


    free(out);
    free(A);
    free(B);
    CHECK_CUDA_ERROR(cudaFree(d_out));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    return 0;
}