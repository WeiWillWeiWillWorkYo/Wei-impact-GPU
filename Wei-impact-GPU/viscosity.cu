#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include "viscosity.cuh"
#include "differences.cuh"

__device__ void computeViscosityForce(double& Fx, double& Fy, double& Fz, double viscosity, double vx1, double vy1, double vz1, double vx2, double vy2, double vz2, double x1, double y1, double z1, double x2, double y2, double z2) {
    double dx, dy, dz, vx, vy, vz;
    computePositionDifferences(dx, dy, dz, x1, y1, z1, x2, y2, z2);
    computeVelocityDifferences(vx, vy, vz, vx1, vy1, vz1, vx2, vy2, vz2);
    double distSq = dx * dx + dy * dy + dz * dz + 1e-10;
    Fx += viscosity * vx / distSq;
    Fy += viscosity * vy / distSq;
    Fz += viscosity * vz / distSq;
}