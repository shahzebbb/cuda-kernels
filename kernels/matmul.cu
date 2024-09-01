#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Function prototypes
__global__ void matrixMultiplication(float* A, float* B, float* C, int M, int N, int K);
void checkCudaError(cudaError_t error, const char *file, int line);

#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__)

// Helper function to check for CUDA errors
void checkCudaError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}

// Host function to setup and launch the kernel
void launchMatrixMultiplication(float* A, float* B, float* C, int M, int N, int K) {
    // Device pointers
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

    // Copy input matrices from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Setup execution configuration
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    matrixMultiplication<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy result matrix from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
}

// Kernel function declaration (you'll implement this)
__global__ void matrixMultiplication(float* A, float* B, float* C, int M, int N, int K) {
    // TODO: Implement the matrix multiplication kernel
}

// Main function (for testing)
int main() {
    // TODO: Add code to test your matrix multiplication
    return 0;
}