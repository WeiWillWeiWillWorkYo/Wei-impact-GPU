#include "datatypes.cuh"

void initializeGrid(Cell& m_cell, int num_particles, int num_cells);
__device__ uint calculateHash(int3 grid_pos, int3 grid_dimensions);
__global__ void fillGrid(Particles p, Cell m_cell, double gridCellSize, int3 gridDimensions);
__global__ void setCellStartsAndEnds(Cell m_cell, int num_particles);
__device__ bool isNeighbor(Particles p, int idx1, int idx2, double radius);
__global__ void findNeighbors(Particles p, Cell m_cell, double radius);