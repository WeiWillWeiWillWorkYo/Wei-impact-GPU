#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include "datatypes.cuh"
#include "integration.cuh"



//5.15 new hash search

void initializeGrid(Cell& m_cell, int num_particles, int num_cells) {
    cudaMallocManaged(&m_cell.sorted_index, num_particles * sizeof(uint));
    cudaMallocManaged(&m_cell.grid_hash, num_particles * sizeof(uint));
    cudaMallocManaged(&m_cell.starts, num_cells * sizeof(uint));
    cudaMallocManaged(&m_cell.ends, num_cells * sizeof(uint));
    m_cell.num_cells = num_cells;

    memset(m_cell.starts, 0xffffffff, num_cells * sizeof(uint));
    memset(m_cell.ends, 0xffffffff, num_cells * sizeof(uint));
}

__device__ uint calculateHash(int3 grid_pos, int3 grid_dimensions) {
    return grid_pos.x + grid_dimensions.x * (grid_pos.y + grid_dimensions.y * grid_pos.z);
}

__global__ void fillGrid(Particles p, Cell m_cell, double gridCellSize, int3 gridDimensions) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= p.count) return;

    int3 grid_pos = make_int3(p.x[idx] / gridCellSize, p.y[idx] / gridCellSize, p.z[idx] / gridCellSize);
    uint hash = calculateHash(grid_pos, gridDimensions);
    m_cell.grid_hash[idx] = hash;
    m_cell.sorted_index[idx] = idx;
}

__global__ void setCellStartsAndEnds(Cell m_cell, int num_particles) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_particles) return;

    if (idx == 0) {
        m_cell.starts[m_cell.grid_hash[0]] = 0;
    }
    else if (m_cell.grid_hash[idx] != m_cell.grid_hash[idx - 1]) {
        m_cell.starts[m_cell.grid_hash[idx]] = idx;
        m_cell.ends[m_cell.grid_hash[idx - 1]] = idx;
    }

    if (idx == num_particles - 1) {
        m_cell.ends[m_cell.grid_hash[idx]] = num_particles;
    }
}

__device__ bool isNeighbor(Particles p, int idx1, int idx2, double radius) {
    double distX = p.x[idx2] - p.x[idx1];
    double distY = p.y[idx2] - p.y[idx1];
    double distZ = p.z[idx2] - p.z[idx1];
    double dist_sq = distX * distX + distY * distY + distZ * distZ;
    double threshold_sq = radius * radius;

    return dist_sq < threshold_sq;
}



/*
//5.10 debug
__global__ void fillGrid(Particles p, double gridCellSize, int3 gridDimensions) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.count) {
        int gridX = int(p.x[i] / gridCellSize);
        int gridY = int(p.y[i] / gridCellSize);
        int gridZ = int(p.z[i] / gridCellSize);

        if (gridX >= 0 && gridX < gridDimensions.x &&
            gridY >= 0 && gridY < gridDimensions.y &&
            gridZ >= 0 && gridZ < gridDimensions.z) {

            int hash = gridX + gridY * gridDimensions.x + gridZ * gridDimensions.x * gridDimensions.y;
            p.hash[i] = hash;
            printf("Particle %d: Position (%.2f, %.2f, %.2f), Grid (%d, %d, %d), Hash %d\n", i, p.x[i], p.y[i], p.z[i], gridX, gridY, gridZ, hash);
        }
    }
}
*/




//4.25 new way to calculate hash

/*
__global__ void fillGrid(Particles p, double gridCellSize, int3 gridDimensions) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.count) {
        int centerOffset = gridDimensions.x / 2;
        int gridX = int(floor(p.x[i] / gridCellSize)) + centerOffset;
        int gridY = int(floor(p.y[i] / gridCellSize)) + centerOffset;
        int gridZ = int(floor(p.z[i] / gridCellSize)) + centerOffset;

        if (gridX >= 0 && gridX < gridDimensions.x &&
            gridY >= 0 && gridY < gridDimensions.y &&
            gridZ >= 0 && gridZ < gridDimensions.z) {


            //4.25 为了解决哈希冲突
            //int prime = 31; // 选择一个素数作为哈希函数中的基数
            //int hash = (gridX * prime * prime) + (gridY * prime) + gridZ;
            int hash = gridX + gridY * gridDimensions.x + gridZ * gridDimensions.x * gridDimensions.y;
            //int hash = gridX + gridY * (gridDimensions.x + 1) + gridZ * (gridDimensions.x + 1) * (gridDimensions.y + 1);

            p.hash[i] = hash;
            printf("Particle %d: Position (%.2f, %.2f, %.2f), Grid (%d, %d, %d), Hash %d\n",i, p.x[i], p.y[i], p.z[i], gridX, gridY, gridZ, hash);
        }
    }
}
*/
/*
__global__ void fillGrid(Particles p, Grid g, double gridCellSize, int3 gridDimensions) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.count) {
        // 偏移中心，以0为中心点
        int centerOffset = gridDimensions.x / 2;

        // 计算每个维度上的网格索引，加上中心偏移量
        int gridX = int(floor(p.x[i] / gridCellSize)) + centerOffset;
        int gridY = int(floor(p.y[i] / gridCellSize)) + centerOffset;
        int gridZ = int(floor(p.z[i] / gridCellSize)) + centerOffset;

        // 确保索引在有效范围内
        if (gridX >= 0 && gridX < gridDimensions.x &&
            gridY >= 0 && gridY < gridDimensions.y &&
            gridZ >= 0 && gridZ < gridDimensions.z) {
            int hash = gridX + gridY * gridDimensions.x + gridZ * gridDimensions.x * gridDimensions.y;
            p.hash[i] = hash;
            if (g.sizes[hash] < MAX_PARTICLES_PER_CELL) {
                int index = atomicAdd(&g.sizes[hash], 1);
                if (index < MAX_PARTICLES_PER_CELL) {
                    g.cells[hash * MAX_PARTICLES_PER_CELL + index] = i;
                    printf("Particle %d assigned to cell %d, position (%.2f, %.2f, %.2f), size now %d\n",
                        i, hash, p.x[i], p.y[i], p.z[i], g.sizes[hash]);
                }
            }
        }
    }
}
*/