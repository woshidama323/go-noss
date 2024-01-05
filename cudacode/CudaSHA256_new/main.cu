// cd /home/hork/cuda-workspace/CudaSHA256/Debug/files
// time ~/Dropbox/FIIT/APS/Projekt/CpuSHA256/a.out -f ../file-list
// time ../CudaSHA256 -f ../file-list


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include "sha256.cuh"
#include <dirent.h>
#include <ctype.h>
#include <cuda_runtime.h>
char * trim(char *str){
    size_t len = 0;
    char *frontp = str;
    char *endp = NULL;

    if( str == NULL ) { return NULL; }
    if( str[0] == '\0' ) { return str; }

    len = strlen(str);
    endp = str + len;

    /* Move the front and back pointers to address the first non-whitespace
     * characters from each end.
     */
    while( isspace((unsigned char) *frontp) ) { ++frontp; }
    if( endp != frontp )
    {
        while( isspace((unsigned char) *(--endp)) && endp != frontp ) {}
    }

    if( str + len - 1 != endp )
            *(endp + 1) = '\0';
    else if( frontp != str &&  endp == frontp )
            *str = '\0';

    /* Shift the string so that it starts at str so that if it's dynamically
     * allocated, we can still free it on the returned pointer.  Note the reuse
     * of endp to mean the front of the string buffer now.
     */
    endp = str;
    if( frontp != str )
    {
            while( *frontp ) { *endp++ = *frontp++; }
            *endp = '\0';
    }


    return str;
}

__global__ void sha256_cuda(JOB ** jobs, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        // perform sha256 calculation here
        if (i < n){
                SHA256_CTX ctx;
                sha256_init(&ctx);
                sha256_update(&ctx, jobs[i]->data, jobs[i]->size);
                sha256_final(&ctx, jobs[i]->digest);
        }
}

void pre_sha256() {
        // compy symbols
        checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}


void runJobs(JOB ** jobs, int n){
        int blockSize = 4;
        int numBlocks = (n + blockSize - 1) / blockSize;
        sha256_cuda <<< numBlocks, blockSize >>> (jobs, n);
}


JOB * JOB_init(BYTE * data, long size, char * fname) {
        JOB * j;
        checkCudaErrors(cudaMallocManaged(&j, sizeof(JOB)));    //j = (JOB *)malloc(sizeof(JOB));
        checkCudaErrors(cudaMallocManaged(&(j->data), size));
        j->data = data;
        j->size = size;
        for (int i = 0; i < 64; i++)
        {
                j->digest[i] = 0xff;
        }
        strcpy(j->fname, fname);
        return j;
}


BYTE * get_file_data(char * fname, unsigned long * size) {
        FILE * f = 0;
        BYTE * buffer = 0;
        unsigned long fsize = 0;

        f = fopen(fname, "rb");
        if (!f){
                fprintf(stderr, "get_file_data Unable to open '%s'\n", fname);
                return 0;
        }
        fflush(f);

        if (fseek(f, 0, SEEK_END)){
                fprintf(stderr, "Unable to fseek %s\n", fname);
                return 0;
        }
        fflush(f);
        fsize = ftell(f);
        rewind(f);

        //buffer = (char *)malloc((fsize+1)*sizeof(char));
        checkCudaErrors(cudaMallocManaged(&buffer, (fsize+1)*sizeof(char)));
        fread(buffer, fsize, 1, f);
        fclose(f);
        *size = fsize;
        return buffer;
}

void print_usage(){
        printf("Usage: CudaSHA256 [OPTION] [FILE]...\n");
        printf("Calculate sha256 hash of given FILEs\n\n");
        printf("OPTIONS:\n");
        printf("\t-f FILE1 \tRead a list of files (separeted by \\n) from FILE1, output hash for each file\n");
        printf("\t-h       \tPrint this help\n");
        printf("\nIf no OPTIONS are supplied, then program reads the content of FILEs and outputs hash for each FILEs \n");
        printf("\nOutput format:\n");
        printf("Hash following by two spaces following by file name (same as sha256sum).\n");
        printf("\nNotes:\n");
        printf("Calculations are performed on GPU, each seperate file is hashed in its own thread\n");
}

extern "C" {
    void hashStringOld(char *str, size_t len, unsigned char *digest) {
        // Convert the string to a BYTE array or use it directly if it's already a BYTE array
        BYTE *byte_str = (BYTE *)str;

        // Initialize JOB and run the hashing job
        JOB *job;
        checkCudaErrors(cudaMallocManaged(&job, sizeof(JOB)));
        job->data = byte_str;
        job->size = len;
        pre_sha256();  // Preprocessing, if required
        runJobs(&job, 1); // Run job
        cudaDeviceSynchronize(); // Synchronize

        // Copy the result to digest
        memcpy(digest, job->digest, 64); // Assuming 64 bytes for SHA-256

        // Cleanup
        cudaFree(job);
    }
}

extern "C" {
    void hashStrings(char **strs, int num_strs, unsigned char **digests) {
        JOB **jobs;
        checkCudaErrors(cudaMallocManaged(&jobs, num_strs * sizeof(JOB *)));

        for (int i = 0; i < num_strs; i++) {
            size_t len = strlen(strs[i]); // Length of the string
            BYTE *byte_str;
            checkCudaErrors(cudaMallocManaged(&byte_str, (len + 1) * sizeof(BYTE)));

	    memcpy(byte_str, strs[i], len);

            jobs[i] = JOB_init(byte_str, len, strs[i]);
        }

        pre_sha256();  // Preprocessing, if required
        runJobs(jobs, num_strs); // Run jobs
        cudaDeviceSynchronize(); // Synchronize

        // Copy the results to digests
        for (int i = 0; i < num_strs; i++) {
             checkCudaErrors(cudaMallocManaged(&(digests[i]), 64 * sizeof(unsigned char))); // Assuming 64 bytes for SHA-256
             memset(digests[i], 0, 64 * sizeof(unsigned char));
	     memcpy(digests[i], jobs[i]->digest, 64);
             // 打印日志信息
             printf("Hash for string %d: ", i);
             for (int j = 0; j < 32; j++) {
                   printf("%02x", digests[i][j]); // 以十六进制格式打印每个字节
             }
    	     printf("\n");
	}

        // Cleanup
        for (int i = 0; i < num_strs; i++) {
            cudaFree(jobs[i]->data);
            cudaFree(jobs[i]);
        }
        cudaFree(jobs);
	for (int i = 0; i < num_strs; i++) {
            printf("Hash %s: ", digests[i]);
            for (int j = 0; j < 64; j++) { // 假设每个哈希值是64字节
                printf("%02x", digests[i][j]); // 以十六进制打印每个字节
            }
            printf("\n");
        }
    }
}

int main(int argc, char **argv) {
    unsigned long temp;
    BYTE *buff;
    JOB **jobs;
    int n = argc - 1; // Number of strings

    if (n > 0) {
        checkCudaErrors(cudaMallocManaged(&jobs, n * sizeof(JOB *)));

        for (int i = 0; i < n; i++) {
            temp = strlen(argv[i + 1]); // Length of the string
            checkCudaErrors(cudaMallocManaged(&buff, (temp + 1) * sizeof(char)));
            memcpy(buff, argv[i + 1], temp);
            jobs[i] = JOB_init(buff, temp, argv[i + 1]);
        }

        pre_sha256();
        runJobs(jobs, n);
        cudaDeviceSynchronize();
        print_jobs(jobs, n);
    }

    cudaDeviceReset();
    return 0;
}
