docker run --runtime=nvidia --gpus all -dti nvidia/cuda:12.3.1-devel-ubuntu22.04 /bin/bash

docker run --runtime=nvidia --rm --name gputest --gpus all -v /data/noss/go-noss/cudacode/hash-shader:/root -dti nvidia/cuda:12.3.1-devel-ubuntu22.04 /bin/bash



docker exec -ti vigilant_knuth /bin/bash


## 坑1 
hash-shader 使用wgpu的rust 库，读不到ubuntu上的GPU