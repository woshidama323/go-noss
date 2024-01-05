extern "C" {
    void hashString(char *str, size_t len, unsigned char *digest) {
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