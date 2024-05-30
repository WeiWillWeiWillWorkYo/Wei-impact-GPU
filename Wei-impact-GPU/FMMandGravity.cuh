#ifndef FMM_H
#define FMM_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstring>

// ç≈ëÂìIó±éqêîó 
#define MAX_PARTICLES_PER_NODE 32

__global__ void buildTree(TreeNode* nodes, Particles particles, int* indices, int numParticles);
void launchBuildTree(TreeNode*& d_nodes, Particles& d_particles, int* d_indices, int numParticles);

__global__ void computeMultipole(TreeNode* nodes, int numNodes);
void launchComputeMultipole(TreeNode* d_nodes, int numNodes);

__global__ void multipoleToMultipole(TreeNode* nodes, int numNodes);
void launchMultipoleToMultipole(TreeNode* d_nodes, int numNodes);

__global__ void computeLocalExpansion(TreeNode* nodes, int numNodes);
void launchComputeLocalExpansion(TreeNode* d_nodes, int numNodes);

__global__ void localToLocal(TreeNode* nodes, int numNodes);
void launchLocalToLocal(TreeNode* d_nodes, int numNodes);

__global__ void computeGravityFMM(TreeNode* nodes, Particles particles, int numNodes);
void launchComputeGravityFMM(TreeNode* d_nodes, Particles& d_particles, int numNodes);

#endif