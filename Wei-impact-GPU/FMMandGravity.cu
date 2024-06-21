#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include "datatypes.cuh"
#include "integration.cuh"
#include "viscosity.cuh"
#include "gravity.cuh"
#include "differences.cuh"
#include "kernel.cuh"
#include "FMMandGravity.cuh"

__global__ void buildTree(TreeNode* nodes, Particles particles, int* indices, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // Initialize root node
    if (idx == 0) {
        nodes[0].start = 0;
        nodes[0].end = numParticles;
        nodes[0].isLeaf = (numParticles <= MAX_PARTICLES_PER_NODE);
        nodes[0].multipole = 0.0;
        for (int i = 0; i < 8; i++) nodes[0].children[i] = -1;

        double totalMass = 0.0;
        double centerX = 0.0, centerY = 0.0, centerZ = 0.0;
        for (int i = 0; i < numParticles; ++i) {
            totalMass += particles.mass[indices[i]];
            centerX += particles.x[indices[i]] * particles.mass[indices[i]];
            centerY += particles.y[indices[i]] * particles.mass[indices[i]];
            centerZ += particles.z[indices[i]] * particles.mass[indices[i]];
        }
        nodes[0].mass = totalMass;
        nodes[0].x = centerX / totalMass;
        nodes[0].y = centerY / totalMass;
        nodes[0].z = centerZ / totalMass;
    }

    __syncthreads();

    // Non-root nodes
    if (idx > 0 && idx < numParticles) {
        // TODO: Implement parallel tree building for non-root nodes
        // This is complex and needs careful handling of memory and parallel tasks
    }
}


void launchBuildTree(TreeNode*& d_nodes, Particles& d_particles, int* d_indices, int numParticles) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    buildTree << <blocksPerGrid, threadsPerBlock >> > (d_nodes, d_particles, d_indices, numParticles);
    cudaDeviceSynchronize();
}

__global__ void computeMultipole(TreeNode* nodes, int numNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    TreeNode* node = &nodes[i];
    if (node->isLeaf) {
        node->multipole = node->mass; // Example: using only monopole term
    }
    else {
        node->multipole = 0.0;
        for (int j = 0; j < 8; ++j) {
            if (node->children[j] != -1) {
                node->multipole += nodes[node->children[j]].multipole;
            }
        }
    }
}

void launchComputeMultipole(TreeNode* d_nodes, int numNodes) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numNodes + threadsPerBlock - 1) / threadsPerBlock;
    computeMultipole << <blocksPerGrid, threadsPerBlock >> > (d_nodes, numNodes);
    cudaDeviceSynchronize();
}


__global__ void multipoleToMultipole(TreeNode* nodes, int numNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    TreeNode* node = &nodes[i];
    for (int j = 0; j < 8; ++j) {
        int childIdx = node->children[j];
        if (childIdx != -1) {
            TreeNode* child = &nodes[childIdx];
            double dx = child->x - node->x;
            double dy = child->y - node->y;
            double dz = child->z - node->z;
            double distance = sqrt(dx * dx + dy * dy + dz * dz);

            // Example: transfer monopole
            child->multipole += node->multipole / distance;
        }
    }
}

void launchMultipoleToMultipole(TreeNode* d_nodes, int numNodes) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numNodes + threadsPerBlock - 1) / threadsPerBlock;
    multipoleToMultipole << <blocksPerGrid, threadsPerBlock >> > (d_nodes, numNodes);
    cudaDeviceSynchronize();
}

__global__ void computeLocalExpansion(TreeNode* nodes, int numNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    TreeNode* node = &nodes[i];
    if (node->isLeaf) {
        node->localExpansion = 0.0; // Initialize local expansion for leaf nodes
    }
    else {
        for (int j = 0; j < 8; ++j) {
            if (node->children[j] != -1) {
                node->localExpansion += nodes[node->children[j]].localExpansion;
            }
        }
    }
}

void launchComputeLocalExpansion(TreeNode* d_nodes, int numNodes) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numNodes + threadsPerBlock - 1) / threadsPerBlock;
    computeLocalExpansion << <blocksPerGrid, threadsPerBlock >> > (d_nodes, numNodes);
    cudaDeviceSynchronize();
}

__global__ void localToLocal(TreeNode* nodes, int numNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    TreeNode* node = &nodes[i];
    for (int j = 0; j < 8; ++j) {
        int childIdx = node->children[j];
        if (childIdx != -1) {
            TreeNode* child = &nodes[childIdx];
            double dx = child->x - node->x;
            double dy = child->y - node->y;
            double dz = child->z - node->z;
            double distance = sqrt(dx * dx + dy * dy + dz * dz);

            // Example: transfer local expansion
            child->localExpansion += node->localExpansion / distance;
        }
    }
}

void launchLocalToLocal(TreeNode* d_nodes, int numNodes) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numNodes + threadsPerBlock - 1) / threadsPerBlock;
    localToLocal << <blocksPerGrid, threadsPerBlock >> > (d_nodes, numNodes);
    cudaDeviceSynchronize();
}

__global__ void computeGravityFMM(TreeNode* nodes, Particles particles, int numNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    TreeNode* node = &nodes[i];
    if (node->isLeaf) {
        for (int j = node->start; j < node->end; ++j) {
            int idx = j;
            particles.ax[idx] += node->localExpansion * particles.mass[idx];
            particles.ay[idx] += node->localExpansion * particles.mass[idx];
            particles.az[idx] += node->localExpansion * particles.mass[idx];
        }
    }
}

void launchComputeGravityFMM(TreeNode* d_nodes, Particles& d_particles, int numNodes) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numNodes + threadsPerBlock - 1) / threadsPerBlock;
    computeGravityFMM << <blocksPerGrid, threadsPerBlock >> > (d_nodes, d_particles, numNodes);
    cudaDeviceSynchronize();
}
