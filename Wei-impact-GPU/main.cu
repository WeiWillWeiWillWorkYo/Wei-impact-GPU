#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "datatypes.cuh"
#include <fstream>
#include <string>
#include <iomanip> // 用于设置输出格式
#include "integration.cuh"
#include "VTKrelated.h"
#include "kernel.cuh"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <random>
#include "initialize.cuh"
#include "Hashsearch.cuh"


int main() {
    const int numParticles = 1000;  // 粒子数量

    const int numParticles1 = 200;
    const int numParticles2 = numParticles - numParticles1;

    const double timeStep = 0.001;  // 时间步长
    const double G = 6.67430e-11;
    const double viscosity = 0.01; // 粘性系数
    const double gridCellSize = 0.08;  // 网格单元的大小
    const int numIterations = 200;  // 模拟总步数
    int3 gridDimensions = make_int3(100, 100, 100);  // 网格的三维尺寸

    std::random_device rd;  // 随机数种子
    std::mt19937 gen(rd());  // 以 rd() 为种子的随机数生成器
    std::uniform_int_distribution<> distrib(0, numParticles - 1);  // 定义范围

    Particles particles;
    cudaError_t status = particles.allocate(numParticles);
    if (status != cudaSuccess) {
        std::cerr << "CUDA error during particles allocation: " << cudaGetErrorString(status) << std::endl;
        return 1;
    }




    // 分配内存给交互对数组
    status = cudaMallocManaged(&(particles.pairs), numParticles * sizeof(InteractionPair));
    if (status != cudaSuccess) {
        std::cerr << "CUDA error during pairs allocation: " << cudaGetErrorString(status) << std::endl;
        return 1;
    }

    // 分配内存给交互对数量
    status = cudaMallocManaged(&(particles.numPairs), numParticles * sizeof(int));
    if (status != cudaSuccess) {
        std::cerr << "CUDA error during numPairs allocation: " << cudaGetErrorString(status) << std::endl;
        return 1;
    }

    *particles.numPairs = 0;  // 初始化交互对数量


    Cell cell;
    int num_cells = gridDimensions.x * gridDimensions.y * gridDimensions.z;
    initializeGrid(cell, numParticles, num_cells);



    int cubeRoot1 = cbrt(numParticles1);  // 计算粒子数量的立方根
    if (cubeRoot1 * cubeRoot1 * cubeRoot1 < numParticles1) {  // 确保有足够空间放置所有粒子
        cubeRoot1++;
    }

    int cubeRoot2 = cbrt(numParticles2);  // 计算粒子数量的立方根
    if (cubeRoot2 * cubeRoot2 * cubeRoot2 < numParticles2) {  // 确保有足够空间放置所有粒子
        cubeRoot2++;
    }

    double spacing = 0.01;  // 粒子间距
    int index = 0;

    particles.count = numParticles; // 初始化粒子数量


    // 以立方体形式初始化粒子

    initialize_particles_sphere(numParticles1, numParticles2, cubeRoot1, cubeRoot2, spacing, particles);



    for (int i = 0; i < numParticles; i++) {
        std::cout << "Particle " << i << ": Position (" << particles.x[i] << ", " << particles.y[i] << ", " << particles.z[i] << ")\n";
    }

    int blockSize = 4;
    int numBlocks = (particles.count + blockSize - 1) / blockSize;

    double h_avg = 0;
    for (int i = 0; i < numParticles; i++) {
        h_avg += particles.h[i];
    }
    h_avg /= numParticles;

    // 在迭代外部初始化
    thrust::device_vector<int> d_hashes(numParticles);
    thrust::device_vector<int> d_indices(numParticles);

    // 粒子半径
    double particlerediu = 0.01;

    // 迭代循环

    
    for (int iter = 0; iter < numIterations; iter++) {
        writeVTKFile(particles, iter);  // 输出VTK文件
        
        thrust::copy(thrust::device, particles.hash, particles.hash + numParticles, d_hashes.begin());
        thrust::sequence(thrust::device, d_indices.begin(), d_indices.end());

        fillGrid <<<numBlocks, blockSize >>> (particles, cell, gridCellSize, gridDimensions);
        cudaDeviceSynchronize();

        thrust::sort_by_key(thrust::device, d_hashes.begin(), d_hashes.end(), d_indices.begin());

        
        setCellStartsAndEnds <<<numBlocks, blockSize >>> (cell, numParticles);
        cudaDeviceSynchronize();

        findNeighbors <<<numBlocks, blockSize >>> (particles, cell, 2 * h_avg);  // 使用2倍的平均光滑核半径作为查找邻居的半径
        cudaDeviceSynchronize();

        // 随机选择一个粒子并找到其邻居
        
       /*
        int selected_index = distrib(gen);  // 生成一个随机粒子索引
        std::vector<int> neighbors;
        for (int i = 0; i < numParticles; i++) {
            if (particles.hash[i] == particles.hash[selected_index] && i != selected_index) {
                double dx = particles.x[i] - particles.x[selected_index];
                double dy = particles.y[i] - particles.y[selected_index];
                double dz = particles.z[i] - particles.z[selected_index];
                double dist = sqrt(dx * dx + dy * dy + dz * dz);

                if (dist < 2 * particles.h[selected_index]) {
                    neighbors.push_back(i);
                }
            }
        }
 
        std::cout << "Selected Particle: " << selected_index << " has " << neighbors.size() << " neighbors.\n";
        for (int idx : neighbors) {
            std::cout << "Neighbor ID: " << idx << " at Position (" << particles.x[idx] << ", " << particles.y[idx] << ", " << particles.z[idx] << ")\n";
        }  */

        // 写入VTK文件
        //writeVTKWithDifferentColors(particles, selected_index, neighbors, iter);
        

       
        //computeInteractions <<<numBlocks, blockSize>>> (particles, timeStep, G);
        //cudaDeviceSynchronize();

        //handleCollisions <<<numBlocks, blockSize>>> (particles, particlerediu);
        //cudaDeviceSynchronize();

        updatePositions <<<numBlocks, blockSize>>> (particles, timeStep);
        cudaDeviceSynchronize();
       
    }

    // 打印更新后的位置
    for (int i = 0; i < numParticles; i++) {
        std::cout << "Particle " << i << ": Position (" << particles.x[i] << ", " << particles.y[i] << ", " << particles.z[i] << ")\n";
        std::cout << "Particle " << i << ": acc (" << particles.ax[i] << ", " << particles.ay[i] << ", " << particles.az[i] << ")\n";
    }

    std::cout << "Average search radius (h_avg): " << h_avg << std::endl;
    std::cout << "Particle spacing: " << spacing << std::endl;
    std::cout << "count: " << particles.count << std::endl;

    particles.free();  // 释放粒子内存

    return 0;
}
