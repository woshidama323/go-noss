# Start from a base image with CUDA
FROM nvidia/cuda:11.0-base

# Install Go
RUN apt-get update && apt-get install -y golang

# Set the working directory
WORKDIR /app

# Copy your Go and CUDA source files
COPY . .

# Compile CUDA code
RUN nvcc -o libhash256.so -shared hash256.cu

# Build your Go application
RUN go build -o myapp main.go

# Run the Go application
CMD ["./myapp"]