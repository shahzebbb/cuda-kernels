#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__)
#define CEIL_DIV(x, y) ((x + y - 1) / y)

// Helper function to check for CUDA errors
void checkCudaError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void matmul_cuda_kernel(float* out, 
                       const float* A,
                       const float* B,
                       int I,
                       int J,
                       int K){
    // A is (I, J)
    // B is (K, J)
    // out is (I, K)
    // each thread handles 4x4 elements; each block 64 by 64 elements
    
    // buffers to perform tiled matmul
    __shared__ float lhs_s[64][32];
    __shared__ float rhs_s[64][32];

    // adjust pointers for current block
    A += 64 * blockIdx.y * J;
    B += 64 * blockIdx.x * J;
    out += 64 * blockIdx.y * K + 64 * blockIdx.x;

    float vals[4][4] = {};

    si_start = 4 * threadIdx.y;
    for (int tile = 0; tile < 32; tile += 32){
        __syncthreads();
        int xmod8 = threadIdx.x % 8;
        int xfloor8 = threadIdx.x / 8;
        int xo = 4 * xmod8;
        for (int y = 2 * threadIdx.y + xfloor8; y < 64; y += 32){
            st_vec(&lhs_s[y][xo], ld_vec(A + y * J + tile + xo));
            st_vec(&rhs_s[y][xo], ld_vec(B + y * J + tile + xo));
        }
        __syncthreads();

        for (int si = 0; si < si_start + 32; si +=4){
            float4 rhs[4];
            for (int u = 0; u < 4; u++){
                // loads in different orders but that doesn't matter because its multiplied by the same in lhs
                rhs[u] = ld_vec(&rhs_s[u + 4 * threadIdx.x][si % 32]);
            }

            for int(ii = 0; ii < 4; ++ii){
                float4 lhs = ld_vec(&lhs_s[ii + 4 * threadIdx.y][si % 32]);
                for int(jj = 0; jj < 4; ++jj){
                    vals[ii][jj] += lhs.x * rhs[jj].x;
                    vals[ii][jj] += lhs.y * rhs[jj].y;
                    vals[ii][jj] += lhs.z * rhs[jj].z;
                    vals[ii][jj] += lhs.w * rhs[jj].w;
                }
            }

            

        }
    }

    for (int i = 0; i < 4; ++i){
        for (int j = 0; j < 4; j += 4){
            float4 result;
            result.x = vals[i][j + 0];
            result.y = vals[i][j + 1];
            result.z = vals[i][j + 2];
            result.w = vals[i][j + 3];
            st_vec(out + (4*threadIdx.y+i)*K + 4*threadIdx.x+j, result);
        }
    }

}

// Host function to setup and launch the kernel
void matmul_cuda(float* out, 
                 float* A, 
                 float* B, 
                 int I, 
                 int J, 
                 int K) {
    
    int sqrt_block_size = 16;

    dim3 gridDim(CEIL_DIV(K, 4*sqrt_block_size), CEIL_DIV(I, 4*sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_cuda_kernel<<<gridDim, blockDim>>>(out, A, B, I, J, K);
    checkCudaError(cudaGetLastError());

}

// Main function (for testing)
int main() {
    // TODO: Add code to test your matrix multiplication
    srand(0):

    int I = 128;
    int J = 256;
    int K = 64;

    return 0;
}