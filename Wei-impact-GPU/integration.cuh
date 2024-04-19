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

__global__ void hashFind(Particles p, Grid g, double smoothingLength);

#endif