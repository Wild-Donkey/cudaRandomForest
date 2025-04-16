// #include <metrics.cuh>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// __device__ entropy_kernel () {}

// float Entropy(int* x, int n) {

// }
// float InfGain(int** x, int *y, int n, int m, int idx);


__global__ void markClass(unsigned* x, unsigned* y, unsigned n, unsigned xtarget) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  y[idx] = (x[idx] == Index) ? 1 : 0;
}

__global__ void computeHistogram(float* data, unsigned int* hist, int dataSize, int numBins, float minVal, float maxVal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dataSize) {
        float val = data[idx];
        int bin = (int)((val - minVal) / (maxVal - minVal) * numBins);
        if (bin >= 0 && bin < numBins) {
            atomicAdd(&hist[bin], 1);
        }
    }
}

// CUDA核函数：计算熵贡献
__global__ void computeEntropyContributions(unsigned int* hist, float* contributions, int numBins, int dataSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBins) {
        unsigned int count = hist[idx];
        if (count > 0) {
            float p = (float)count / dataSize;
            contributions[idx] = -p * log2f(p);
        } else {
            contributions[idx] = 0.0f;
        }
    }
}

// 计算数组的信息熵
float computeEntropy(float* x, int n, int class_count) {
    unsigned *d_x, *d_hist;
    float entropy = 0.0f;
    
    unsigned threadsPerBlock = 256;
    unsigned n_block = (n + threadsPerBlock - 1) / threadsPerBlock;
    unsigned c_block = (class_count + threadsPerBlock - 1) / threadsPerBlock;

    // 1. 分配设备内存
    cudaMalloc((void**)&d_x, n_block * threadsPerBlock * sizeof(unsigned));
    cudaMalloc((void**)&d_hist, c_block * threadsPerBlock * sizeof(unsigned));
    
    cudaMemcpy(d_x, h_data, n_block * threadsPerBlock * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, c_block * threadsPerBlock * sizeof(unsigned int));

    
    if (numBins <= 3) {
      unsigned *d_mark;
      cudaMalloc((void**)&d_mark, n * sizeof(unsigned));
      for (int i = 0; i < numBins; i++) {
        markClass<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_mark, n, i);
         = ;
        cub::DeviceReduce::Sum(d_hist + i, sizeof(unsigned), d_mark, &totalSum, n);
      }
      cudaFree(d_mark);
    } else {
      computeHistogram<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_hist, dataSize, numBins, minVal, maxVal);
    }
    
    // 4. 计算数据最小最大值 (这里简化处理，实际应用中可能需要单独计算)
    float minVal = 0.0f; // 应根据实际数据调整
    float maxVal = 1.0f; // 应根据实际数据调整
    
    // 5. 计算直方图
    computeHistogram<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_hist, dataSize, numBins, minVal, maxVal);
    
    // 6. 计算每个bin的熵贡献
    computeEntropyContributions<<<(numBins + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>
        (d_hist, d_contributions, numBins, dataSize);
    
    // 7. 拷贝贡献值回主机
    cudaMemcpy(h_contributions, d_contributions, numBins * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 8. 计算总熵
    for (int i = 0; i < numBins; i++) {
        entropy += h_contributions[i];
    }
    
    // 9. 释放内存
    cudaFree(d_data);
    cudaFree(d_hist);
    cudaFree(d_contributions);
    free(h_contributions);
    
    return entropy;
}

int main() {
    // 示例数据
    const int dataSize = 1000000;
    const int numBins = 256; // 直方图bin数量
    
    float *h_data = (float*)malloc(dataSize * sizeof(float));
    
    // 生成随机数据 (0-1之间)
    for (int i = 0; i < dataSize; i++) {
        h_data[i] = (float)rand() / RAND_MAX;
    }
    
    // 计算熵
    float entropy = computeEntropy(h_data, dataSize, numBins);
    printf("信息熵: %f\n", entropy);
    
    free(h_data);
    return 0;
}