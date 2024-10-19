#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "common.h"

__global__ void matmul_forward_kernel(float* out, float* A, float* B, 
                               int I, int J, int K){
    // out = A * B.T
    // A is of size (I,J)
    // B is of size (K,J)
    // out will be of size (I,K)
    // simple matmul forward kernel

    int A_row = blockDim.y * blockIdx.y + threadIdx.y;
    int B_col = blockDim.x * blockIdx.x + threadIdx.x;

    if (A_row < I && B_col < K){
        float val = 0.0f;
        for (int i = 0; i < J; i++){
            val += A[A_row * J + i] * B[B_col * J + i];
        }
        out[K * A_row + B_col] = val;
    } 
}

void matmul_forward(float* out, float* A, float* B, 
                    int I, int J, int K, int sqrt_block_size){
    
    dim3 gridDim(CEIL_DIV(K, sqrt_block_size), CEIL_DIV(I, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_kernel<<<gridDim, blockDim>>>(out, A, B, I, J, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
    }

int main(){
    srand(0);

    int I = 512;
    int J = 256;
    int K = 256;

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

    int sqrt_block_sizes[] = {4, 8, 16, 32};

    for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++){
        int sqrt_block_size = sqrt_block_sizes[j];
        printf("Checking block size %d x %d.\n", sqrt_block_size, sqrt_block_size);
        matmul_forward(d_out, d_A, d_B, I, J, K, sqrt_block_size);
        validate_result(d_out, out, I * K, 1e-4);
    }
    for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++){
        int sqrt_block_size = sqrt_block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, matmul_forward,
                                              d_out, d_A, d_B, I, J, K, sqrt_block_size);
        
        float tflops = (float)I * (2*J - 1) * K  / elapsed_time * 1e3f / 1e12f;
        printf("sqrt_block_size %4d | time %.4f ms | tflops %.2f \n", sqrt_block_size, elapsed_time, tflops);
    }

    free(out);
    free(A);
    free(B);
    CHECK_CUDA_ERROR(cudaFree(d_out));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    return 0;
}