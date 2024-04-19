#pragma once
#ifndef VISCOSITY_CUH
#define VISCOSITY_CUH


__device__ void computeViscosityForce(double& Fx, double& Fy, double& Fz, double viscosity, double vx1, double vy1, double vz1, double vx2, double vy2, double vz2, double x1, double y1, double z1, double x2, double y2, double z2);

#endif