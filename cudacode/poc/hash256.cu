#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "sha256.cuh"

__global__ void hash256(unsigned char *input, unsigned char *output, int num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements) {
        SHA256(&input[index * 10], 10, &output[index * SHA256_DIGEST_LENGTH]);
    }
}

extern "C" void runHash256(unsigned char *input, unsigned char *output, int num_elements) {
    unsigned char *d_input, *d_output;
    cudaMalloc((void **)&d_input, num_elements * 10 * sizeof(unsigned char));
    cudaMalloc((void **)&d_output, num_elements * SHA256_DIGEST_LENGTH * sizeof(unsigned char));

    cudaMemcpy(d_input, input, num_elements * 10 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    hash256<<<1, num_elements>>>(d_input, d_output, num_elements);

    cudaMemcpy(output, d_output, num_elements * SHA256_DIGEST_LENGTH * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}