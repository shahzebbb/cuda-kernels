#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>


#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__) // for error checking
#define CEIL_DIV(x, y) ((x + y - 1) / y)

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

