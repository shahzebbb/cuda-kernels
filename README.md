# cuda-kernels
A codebase containing different cuda kernel implementations useful for machine learning. I built this repo for educational purposes. The cuda kernels implemented are:

* `kernels/matmul_llmc.cu`: Contains an implementation of matmul as written in llmc. This implementation is INCOMPLETE and should NOT be used.
* `kernels/matmul_simple.cu`: Contains an implementation of a very simple matmul kernel.
* `kernels/matmul_tiled.cu`: Contains an implementation of tiled matmul.

## Usage
### Prerequisite
I used the docker image `nvidia/cuda:11.7.1-devel-ubuntu22.04` to make this codebase.

### Benchmark
To run any kernel running the following commands, replace <example-kernel> with the kernel of your choice:
```
nvcc -o kernels/<example-kernel> kernels/<example-kernel>.cu
./kernels/<example-kernel>
```

This will perform a check with the cpu version of matmul (obtained from llmc) to validate if the kernel is working correctly. Then it will calculate the TFLOPs of the kernel.

## Todos
- [ ] Add a softmax forward kernel
- [ ] Add a self-attention forward kernel
- [ ] Add a flash attention forward kernel
