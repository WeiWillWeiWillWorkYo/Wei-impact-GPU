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

// 新增：hashFind全局函数

__global__ void hashFind(Particles p, Grid g, double smoothingLength) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p.count) {
        return;
    }

    int hash = p.hash[idx];
    int pairIndex = 0;

    for (int i = 0; i < g.sizes[hash]; i++) {
        int j = g.cells[hash * MAX_PARTICLES_PER_CELL + i];
        if (idx != j) {
            double distX = p.x[j] - p.x[idx];
            double distY = p.y[j] - p.y[idx];
            double distZ = p.z[j] - p.z[idx];
            double m_dist = sqrt(distX * distX + distY * distY + distZ * distZ);
            double h_avg = (p.h[idx] + p.h[j]) / 2.0;

            if (m_dist < 2 * h_avg) {
                double3 dist = make_double3(distX, distY, distZ);
                double3 grad;
                double weight;
                kernel(m_dist, dist, weight, grad, h_avg); // Corrected kernel function call

                if (pairIndex < MAX_PAIRS) {
                    p.pairs[pairIndex].i = idx;
                    p.pairs[pairIndex].j = j;
                    p.pairs[pairIndex].w = weight;
                    p.pairs[pairIndex].grad = grad;
                    pairIndex++;
                }
            }
        }
    }
    *p.numPairs = pairIndex;
    if (threadIdx.x == 0) {
        printf("Total pairs found: %d\n", *p.numPairs);
    }
}
/*
__global__ void hashFind(Particles p, Grid g, double smoothingLength) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p.count)
    {
        printf("Gocha!");
        return;
    }

    int hash = p.hash[idx];
    int pairIndex = 0;

    for (int i = 0; i < g.sizes[hash]; i++) {
        int j = g.cells[hash * MAX_PARTICLES_PER_CELL + i];
       
        if (idx != j) {
            double distX = p.x[j] - p.x[idx];
            double distY = p.y[j] - p.y[idx];
            double distZ = p.z[j] - p.z[idx];
            double m_dist = sqrt(distX * distX + distY * distY + distZ * distZ);
            double h_avg = (p.h[idx] + p.h[j]) / 2.0;
    
            if (m_dist < 2 * h_avg) {
                double3 dist = make_double3(distX, distY, distZ);
                double3 grad;
                double weight;
                kernel(m_dist, dist, weight, grad, h_avg); // Corrected kernel function call

                if (pairIndex < MAX_PAIRS) {
                    p.pairs[pairIndex].i = idx;
                    p.pairs[pairIndex].j = j;
                    p.pairs[pairIndex].w = weight;
                    p.pairs[pairIndex].grad = grad;
                    pairIndex++;
                }
            }
        }
    }
    p.numPairs = pairIndex;
    if (threadIdx.x == 0) {
        printf("Total pairs found: %d\n", p.numPairs);
    }
}
*/
/*
__global__ void computeInteractions(Particles p, double dt, double G) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < *p.numPairs) {
        printf("GGGGG!!!\n");
        InteractionPair pair = p.pairs[idx];
        double Fx = 0, Fy = 0, Fz = 0;

        computeGravitationalForce(Fx, Fy, Fz, G, p.mass[pair.i], p.mass[pair.j], p.x[pair.i], p.y[pair.i], p.z[pair.i], p.x[pair.j], p.y[pair.j], p.z[pair.j]);
        // 更新粒子i和粒子j的速度

        double ax_i = Fx / p.mass[pair.i];
        double ay_i = Fy / p.mass[pair.i];
        double az_i = Fz / p.mass[pair.i];

        // 更新粒子j的加速度（作用和反作用）
        double ax_j = -Fx / p.mass[pair.j];
        double ay_j = -Fy / p.mass[pair.j];
        double az_j = -Fz / p.mass[pair.j];

        // 根据加速度更新速度
        p.vx[pair.i] += ax_i * dt;
        p.vy[pair.i] += ay_i * dt;
        p.vz[pair.i] += az_i * dt;

        p.vx[pair.j] += ax_j * dt;
        p.vy[pair.j] += ay_j * dt;
        p.vz[pair.j] += az_j * dt;
    }
}
*/

/*
__global__ void computeInteractions(Particles p, double dt, double G) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < *p.numPairs) {
        InteractionPair pair = p.pairs[idx];
        double Fx = 0, Fy = 0, Fz = 0;
        double pressure_force_x = 0, pressure_force_y = 0, pressure_force_z = 0;

        double dx = p.x[pair.j] - p.x[pair.i];
        double dy = p.y[pair.j] - p.y[pair.i];
        double dz = p.z[pair.j] - p.z[pair.i];
        double dist = sqrt(dx * dx + dy * dy + dz * dz);
        double h_avg = (p.h[pair.i] + p.h[pair.j]) * 0.5;

        // 计算权重和梯度
        double weight;
        double3 grad;
        kernel(dist, make_double3(dx, dy, dz), weight, grad, h_avg);

        // 计算由压力导致的力
        double pressure_term = (p.p[pair.i] / (p.rho[pair.i] * p.rho[pair.i]) + p.p[pair.j] / (p.rho[pair.j] * p.rho[pair.j])) * p.mass[pair.j];
        pressure_force_x += -pressure_term * grad.x;
        pressure_force_y += -pressure_term * grad.y;
        pressure_force_z += -pressure_term * grad.z;

        // 引力计算
        computeGravitationalForce(Fx, Fy, Fz, G, p.mass[pair.i], p.mass[pair.j], p.x[pair.i], p.y[pair.i], p.z[pair.i], p.x[pair.j], p.y[pair.j], p.z[pair.j]);

        // 综合引力和压力更新速度
        double ax_i = (Fx + pressure_force_x) / p.mass[pair.i];
        double ay_i = (Fy + pressure_force_y) / p.mass[pair.i];
        double az_i = (Fz + pressure_force_z) / p.mass[pair.i];

        // 作用与反作用
        double ax_j = -(Fx + pressure_force_x) / p.mass[pair.j];
        double ay_j = -(Fy + pressure_force_y) / p.mass[pair.j];
        double az_j = -(Fz + pressure_force_z) / p.mass[pair.j];

        p.vx[pair.i] += ax_i * dt;
        p.vy[pair.i] += ay_i * dt;
        p.vz[pair.i] += az_i * dt;

        p.vx[pair.j] += ax_j * dt;
        p.vy[pair.j] += ay_j * dt;
        p.vz[pair.j] += az_j * dt;
    }
}

*/

__global__ void updatePositions(Particles p, double dt) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < p.count) {
            p.x[i] += p.vx[i] * dt; // 更新 x 位置
            p.y[i] += p.vy[i] * dt; // 更新 y 位置
            p.z[i] += p.vz[i] * dt; // 更新 z 位置

        }
    }
