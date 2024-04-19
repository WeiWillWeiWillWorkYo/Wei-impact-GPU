#ifndef DIFFERENCES_CUH
#define DIFFERENCES_CUH
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "datatypes.cuh"

__device__ void computePositionDifferences(double& dx, double& dy, double& dz, const double x1, const double y1, const double z1, const double x2, const double y2, const double z2);

__device__ void computeVelocityDifferences(double& vx, double& vy, double& vz, const double vx1, const double vy1, const double vz1, const double vx2, const double vy2, const double vz2);

#endif