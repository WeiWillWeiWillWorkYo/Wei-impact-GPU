#pragma once
#ifndef DATATYPES_CUH
#define DATATYPES_CUH

#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include "kernel.cuh"


const int MAX_PARTICLES_PER_CELL = 10000; // 假设每个网格最多100个粒子
const int MAX_PAIRS = 1000;            // 假设最大交互对数量

typedef unsigned int uint;

struct InteractionPair {
    int i, j;             // 粒子索引对
    double w;             // 核函数权重
    double3 grad;         // 梯度
};

struct Particles {
    // 基础空间、运动和物理属性
    double* x; double* y; double* z;        // Position
    double* vx; double* vy; double* vz;     // Speed
    double* ax; double* ay; double* az;     // Acceleration
    double* v_prev_x; double* v_prev_y; double* v_prev_z;  // Previous speed
    double* av_x; double* av_y; double* av_z;              // Average speed

    double* mass;                          // Mass
    double* p;                             // Pressure
    double* rho;                           // Density
    double* rho_prev;                      // Previous density
    double* e;                             // Energy
    double* e_prev;                        // Previous energy
    double* drhodt;                        // Density variation
    double* dedt;                          // Energy variation
    double* c;                             // Sound speed
    double* h;                             // Smoothing size

    // 张量属性
    double* epsilon;   // Strain rate tensor
    double* tau;       // Shear stress tensor
    double* tau_dot;   // Shear stress rate tensor
    double* sigma;     // Total stress tensor
    double* tau_prev;  // Previous shear stress tensor
    double* R;         // Additional tensor (purpose needs clarification)

    short* virt;       // Virtual particle flag
    InteractionPair* pairs; // 交互对数组
    int* numPairs;           // 交互对的数量

    int* hash;         // 哈希值对应的网格单元索引
    int count;
    cudaError_t Particles::allocate(int n) {
        count = n;
        cudaError_t status;

        // 基本物理和空间属性的内存分配
        status = cudaMallocManaged(&x, n * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&y, n * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&z, n * sizeof(double));
        if (status != cudaSuccess) return status;

        status = cudaMallocManaged(&vx, n * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&vy, n * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&vz, n * sizeof(double));
        if (status != cudaSuccess) return status;

        status = cudaMallocManaged(&ax, n * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&ay, n * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&az, n * sizeof(double));
        if (status != cudaSuccess) return status;

        status = cudaMallocManaged(&v_prev_x, n * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&v_prev_y, n * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&v_prev_z, n * sizeof(double));
        if (status != cudaSuccess) return status;

        status = cudaMallocManaged(&av_x, n * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&av_y, n * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&av_z, n * sizeof(double));
        if (status != cudaSuccess) return status;

        // 其他动态属性
        status = cudaMallocManaged(&mass, n * sizeof(double));
        if (status != cudaSuccess) return status;

        status = cudaMallocManaged(&p, n * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&rho, n * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&e, n * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&c, n * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&h, n * sizeof(double));
        if (status != cudaSuccess) return status;

        status = cudaMallocManaged(&rho_prev, n * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&e_prev, n * sizeof(double));
        if (status != cudaSuccess) return status;

        status = cudaMallocManaged(&drhodt, n * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&dedt, n * sizeof(double));
        if (status != cudaSuccess) return status;

        status = cudaMallocManaged(&hash, n * sizeof(int));
        if (status != cudaSuccess) return status;

        status = cudaMallocManaged(&virt, n * sizeof(short));
        if (status != cudaSuccess) return status;

        // 张量属性的内存分配
        status = cudaMallocManaged(&epsilon, n * 9 * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&tau, n * 9 * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&sigma, n * 9 * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&tau_prev, n * 9 * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&pairs, MAX_PAIRS * sizeof(InteractionPair));
        if (status != cudaSuccess) return status;
        return cudaSuccess;
    }



    void free() {
        // Free all allocated memory
        cudaFree(x); cudaFree(y); cudaFree(z);
        cudaFree(vx); cudaFree(vy); cudaFree(vz);
        cudaFree(ax); cudaFree(ay); cudaFree(az);
        cudaFree(v_prev_x); cudaFree(v_prev_y); cudaFree(v_prev_z);
        cudaFree(av_x); cudaFree(av_y); cudaFree(av_z);

        cudaFree(mass);
        cudaFree(p);
        cudaFree(rho);
        cudaFree(rho_prev);
        cudaFree(e);
        cudaFree(e_prev);
        cudaFree(drhodt);
        cudaFree(dedt);
        cudaFree(c);
        cudaFree(h);

        cudaFree(epsilon);
        cudaFree(tau);
        cudaFree(tau_dot);
        cudaFree(sigma);
        cudaFree(tau_prev);
        cudaFree(R);

        cudaFree(virt);

        cudaFree(hash);
        cudaFree(pairs);
    }
};

struct Cell {

uint* sorted_index;
uint* grid_hash;
int* starts;
int* ends;
int num_cells;

// Allocation function
cudaError_t allocate(int numParticles, int numCells) {
    this->num_cells = numCells;
    cudaError_t err;
    err = cudaMallocManaged(&sorted_index, numParticles * sizeof(uint));
    if (err != cudaSuccess) return err;
    err = cudaMallocManaged(&grid_hash, numParticles * sizeof(uint));
    if (err != cudaSuccess) return err;
    err = cudaMallocManaged(&starts, numCells * sizeof(int));
    if (err != cudaSuccess) return err;
    err = cudaMallocManaged(&ends, numCells * sizeof(int));
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}

struct TreeNode {
    double x, y, z; // Center of mass
    double mass;    // Total mass
    int start; // Starting index of particles in this node
    int end;   // Ending index of particles in this node
    int children[8]; // Child nodes
    bool isLeaf;    // Flag to indicate if the node is a leaf

    // Multipole expansion coefficients (example: only monopole for simplicity)
    double multipole;
    // Local expansion coefficients
    double localExpansion;
};


// Free function
void free() {
    cudaFree(sorted_index);
    cudaFree(grid_hash);
    cudaFree(starts);
    cudaFree(ends);
}
};

/*
struct Grid {
    int* cells; // 网格中的粒子索引
    int* sizes; // 每个网格存储的粒子数量
    int3 dimensions; // 网格的三维尺寸

    cudaError_t allocate(int3 gridDimensions) {
        dimensions = gridDimensions;
        int totalCells = dimensions.x * dimensions.y * dimensions.z;
        cudaError_t status;
        status = cudaMallocManaged(&cells, totalCells * MAX_PARTICLES_PER_CELL * sizeof(int));
        if (status != cudaSuccess) return status;
        status = cudaMallocManaged(&sizes, totalCells * sizeof(int));
        if (status != cudaSuccess) return status;
        // 使用 cudaMemset 初始化 sizes 数组
        status = cudaMemset(sizes, 0, totalCells * sizeof(int));
        return status;
    }

    void free() {
        cudaFree(cells);
        cudaFree(sizes);
    }
};
*/
void initializeGrid(Cell& m_cell, int num_particles, int num_cells);
__device__ uint calculateHash(int3 grid_pos, int3 grid_dimensions);
__global__ void fillGrid(Particles p, Cell m_cell, double gridCellSize, int3 gridDimensions);
__global__ void setCellStartsAndEnds(Cell m_cell, int num_particles);
__device__ bool isNeighbor(Particles p, int idx1, int idx2, double radius);



//__global__ void fillGrid(Particles p, double gridCellSize, int3 gridDimensions);

#endif