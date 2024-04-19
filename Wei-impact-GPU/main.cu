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



int main() {
    const int numParticles = 1000;  // 粒子数量
    const double timeStep = 0.001;  // 时间步长
    const double G = 6.67430e-11;  // 万有引力常数
    //const double G = 6.67430e-3;
    const double viscosity = 0.01; //粘性系数
    const double gridCellSize = 0.2;  // 网格单元的大小
    const int numIterations = 100;  // 模拟总步数
    int3 gridDimensions = make_int3(30, 30, 30);  // 网格的三维尺寸
    
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
    status = cudaMallocManaged(&(particles.numPairs), sizeof(int));
    if (status != cudaSuccess) {
        std::cerr << "CUDA error during numPairs allocation: " << cudaGetErrorString(status) << std::endl;
        return 1;
    }

    *particles.numPairs = 0;  // 初始化交互对数量


    Grid grid;
    status = grid.allocate(gridDimensions);
    if (status != cudaSuccess) {
        std::cerr << "CUDA error during allocation: " << cudaGetErrorString(status) << std::endl;
        particles.free();  // 释放粒子内存
        return 1;
    }

    int cubeRoot = cbrt(numParticles); // 计算粒子数量的立方根
    if (cubeRoot * cubeRoot * cubeRoot < numParticles) { // 确保有足够空间放置所有粒子
        cubeRoot++;
    }
    double spacing = 0.1; // 粒子间距
    int index = 0;

    // 以立方体形式初始化粒子
    for (int z = 0; z < cubeRoot; z++) {
        for (int y = 0; y < cubeRoot; y++) {
            for (int x = 0; x < cubeRoot; x++) {
                if (index < numParticles) {
                    // 位置和速度初始化
                    particles.x[index] = spacing * x;
                    particles.y[index] = spacing * y;
                    particles.z[index] = spacing * z;
                    particles.vx[index] = 0.0;
                    particles.vy[index] = 0.0;
                    particles.vz[index] = 0.0;
                    particles.mass[index] = 1000.0; // 假设所有粒子的质量为1

                    // 静态物理量初始化
                    particles.p[index] = 101325; // 假定初始压力为大气压
                    particles.rho[index] = 1000; // 假定初始密度，例如水的密度
                    particles.e[index] = 0; // 初始能量设为0
                    particles.c[index] = 343; // 假定初始声速，例如空气中的声速
                    particles.rho_prev[index] = 0; // 上一时刻的密度
                    particles.e_prev[index] = 0; // 上一时刻的能量
                    particles.h[index] = 0.1; // 光滑核半径

                    // 动态物理量初始化
                    particles.drhodt[index] = 0; // 密度变化率初始为0
                    particles.dedt[index] = 0; // 能量变化率初始为0
                    particles.av_x[index] = 0; // 平均速度初始化为0
                    particles.av_y[index] = 0;
                    particles.av_z[index] = 0;

                    // 张量初始化
                    for (int d = 0; d < 9; d++) {
                        particles.epsilon[index * 9 + d] = 0;
                        particles.tau[index * 9 + d] = 0;
                        particles.sigma[index * 9 + d] = 0;
                        particles.tau_prev[index * 9 + d] = 0;
                    }

                    // 其他初始化
                    particles.virt[index] = 0; // 假定所有粒子最初都不是虚拟粒子

                    index++;
                }
            }
        }
    }


    for (int i = 0; i < numParticles; i++) {
        std::cout << "Particle " << i << ": Position (" << particles.x[i] << ", " << particles.y[i] << ", " << particles.z[i] << ")\n";
    }

    int blockSize = 1;
    int numBlocks = (particles.count + blockSize - 1) / blockSize;

    // 为指针分配内存


    double h_avg = 0;
    for (int i = 0; i < numParticles; i++) {
        h_avg += particles.h[i];
    }
    h_avg /= numParticles;



    for (int iter = 0; iter < numIterations; iter++) {
        writeVTKFile(particles, iter);  // 输出VTK文件

        cudaMemset(grid.sizes, 0, gridDimensions.x * gridDimensions.y * gridDimensions.z * sizeof(int));

        fillGrid <<<numBlocks, blockSize >>> (particles, grid, gridCellSize, gridDimensions);
        if (status != cudaSuccess) {
            std::cerr << "CUDA error after HashFind: " << cudaGetErrorString(status) << std::endl;
            break;
        }
        cudaDeviceSynchronize();


        hashFind << <numBlocks, blockSize >> > (particles, grid, h_avg);

        cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            std::cerr << "CUDA error after fillGrid: " << cudaGetErrorString(status) << std::endl;
            break;
        }

        computeInteractions << <numBlocks, blockSize >> > (particles, timeStep, G);
        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            std::cerr << "CUDA error after computeInteractions: " << cudaGetErrorString(status) << std::endl;
            break;
        }
        /**/
        updatePositions << <numBlocks, blockSize >> > (particles, timeStep);
        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            std::cerr << "CUDA error after updatePositions: " << cudaGetErrorString(status) << std::endl;
            break;
        }
    }

    // 打印更新后的位置
    for (int i = 0; i < numParticles; i++) {
        std::cout << "Particle " << i << ": Position (" << particles.x[i] << ", " << particles.y[i] << ", " << particles.z[i] << ")\n";
    }
    std::cout << "Average search radius (h_avg): " << h_avg << std::endl;
    std::cout << "Particle spacing: " << spacing << std::endl;
    std::cout << "count: " << particles.count << std::endl;

    

    particles.free();  // 释放粒子内存
    grid.free();  // 释放网格内存



    return 0;
}