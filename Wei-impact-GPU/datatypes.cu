#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include "datatypes.cuh"
#include "integration.cuh"

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
