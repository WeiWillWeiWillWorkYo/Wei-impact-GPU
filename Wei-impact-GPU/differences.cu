#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "datatypes.cuh"
#include "differences.cuh"

__device__ void computePositionDifferences(double& dx, double& dy, double& dz, const double x1, const double y1, const double z1, const double x2, const double y2, const double z2) {
    dx = x2 - x1;
    dy = y2 - y1;
    dz = z2 - z1;
}

__device__ void computeVelocityDifferences(double& vx, double& vy, double& vz, const double vx1, const double vy1, const double vz1, const double vx2, const double vy2, const double vz2) {
    vx = vx2 - vx1;
    vy = vy2 - vy1;
    vz = vz2 - vz1;
}