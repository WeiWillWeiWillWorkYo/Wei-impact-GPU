#include "Hashsearch.cuh"
#include "datatypes.cuh"

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

__global__ void findNeighbors(Particles p, Cell m_cell, double radius) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= p.count) return;

    int start_idx = m_cell.starts[m_cell.grid_hash[idx]];
    int end_idx = m_cell.ends[m_cell.grid_hash[idx]];
    int pairIndex = 0;

    // 确保 start_idx 和 end_idx 在有效范围内
    if (start_idx == -1 || end_idx == -1 || start_idx >= end_idx) return;

    for (int j = start_idx; j < end_idx; ++j) {
        int neighbor_idx = m_cell.sorted_index[j];
        if (neighbor_idx != idx && isNeighbor(p, idx, neighbor_idx, radius)) {
            double distX = p.x[neighbor_idx] - p.x[idx];
            double distY = p.y[neighbor_idx] - p.y[idx];
            double distZ = p.z[neighbor_idx] - p.z[idx];
            double dist = sqrt(distX * distX + distY * distY + distZ * distZ);
            double3 dist_vec = make_double3(distX, distY, distZ);
            double3 grad;
            double weight;
            double h_avg = (p.h[idx] + p.h[neighbor_idx]) / 2.0;
            //printf("Particle %d interacting with %d: dist = %f, h_avg = %f\n", idx, neighbor_idx, dist, h_avg);

            // 计算权重和梯度
            kernel(dist, dist_vec, weight, grad, h_avg);

            //printf("Weight: %f, Grad: (%f, %f, %f)\n", weight, grad.x, grad.y, grad.z);
            // 更新粒子属性
            if (pairIndex < MAX_PAIRS) {
                p.pairs[pairIndex].i = idx;
                p.pairs[pairIndex].j = neighbor_idx;
                p.pairs[pairIndex].w = weight;
                p.pairs[pairIndex].grad = grad;
                pairIndex++;
            }
        }
    }
    p.numPairs[idx] = pairIndex; // 更新每个粒子的配对数

}