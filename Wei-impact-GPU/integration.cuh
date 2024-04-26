#pragma once
#ifndef INTERGRATION_CUH
#define INTERGRATION_CUH

#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include "datatypes.cuh"
//#include "viscosity.cuh"
//#include "gravity.cuh"

__global__ void computeInteractions(Particles p, double dt, double G);

__global__ void updatePositions(Particles p, double dt);


//4.25 new hash find
__global__ void hashFind(Particles p, int numParticles);

__global__ void handleCollisions(Particles p, double radius);

__device__ void handleCollision(double3& vel_i, double3& vel_j, double3 pos_i, double3 pos_j, double mass_i, double mass_j);

//__global__ void hashFind(Particles p, Grid g, double smoothingLength);

#endif