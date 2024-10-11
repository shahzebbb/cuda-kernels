#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__) // for error checking
#define CEIL_DIV(x, y) ((x + y - 1) / y)

// Helper function to check for CUDA errors
void checkCudaError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}

float* make_random_float(size_t N){
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++){
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    return arr;
}

void validate_result(float* device_result, float* cpu_reference, int num_elements, float tolerance=1e-4){
    float* out_gpu = (float*)malloc(num_elements * sizeof(float));
    CHECK_CUDA_ERROR(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(float), cudaMemcpyDeviceToHost));
    int nfaults = 0;
    for (int i = 0; i < num_elements; i++){
        if (i < 5){
            printf("%f %f\n", cpu_reference[i], out_gpu[i]);
        }

        if ((cpu_reference[i] - out_gpu[i]) > tolerance) {
            printf("Mismatch at %d: CPU_ref: %f vs GPU: %f\n", i, cpu_reference[i], out_gpu[i]);
            nfaults ++;
            if (nfaults > 10){
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }
    if (nfaults > 0){
        free(out_gpu);
        exit(EXIT_FAILURE);
    }

    free(out_gpu);
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args){
    cudaEvent_t start, stop;
    int deviceIdx = 0;
    CHECK_CUDA_ERROR(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    void* flush_buffer;
    CHECK_CUDA_ERROR(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    float elapsed_time = 0.0f;
    for (int i = 0; i < repeats; i++){
        CHECK_CUDA_ERROR(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        CHECK_CUDA_ERROR(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        CHECK_CUDA_ERROR(cudaEventRecord(stop, nullptr));
        CHECK_CUDA_ERROR(cudaEventSynchronize(start));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        float single_call;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    CHECK_CUDA_ERROR(cudaFree(flush_buffer));

    return elapsed_time / repeats;

}

void matmul_forward_cpu(float* out,
                    const float* A, const float* B,
                    int I, int J, int K) {
    // out = A * B.T
    // A is of size (I,J)
    // B is of size (K,J)
    // out will be of size (I,K)

    for (int i = 0; i < I; i++) {
        float* out_i = out + i * K;
        const float* a_i = A + i * J;
        for (int k = 0; k < K; k++) {
            float val = 0.0f;
            const float* bcol = B + k*J;
            for (int j = 0; j < J; j++) {
                val += a_i[j] * bcol[j];
            }
            out_i[k] = val;
        }
    }
}

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
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * K * sizeof(float)));

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