# CudaSHA256
Simple tool to calculate sha256 on GPU using Cuda

# Built
```
nvcc main.cu
```

# Run
```
./a.out <some test file> <another test file> ...
or
nvprof ./a.out <some test file> <another test file> ...
```

### build
```
sudo nvcc  -Xcompiler -fPIC  -o libcuda_hash.so  --shared main.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
```