#include <cmath>
#include <cstdlib> // For rand()
#include "datatypes.cuh"


double M_PI = 3.1415926;
// Helper function to generate random double between min and max
double random_double(double min, double max) {
    return min + (max - min) * (rand() / (double)RAND_MAX);
}
// Calculate the radius of a sphere based on the number of particles and their spacing


void initialize_particles_sphere(int numParticles1, int numParticles2, int cubeRoot1, int cubeRoot2, double spacing, Particles& particles) {
    int index = 0;

    for (int z = 0; z < cubeRoot1; z++) {
        for (int y = 0; y < cubeRoot1; y++) {
            for (int x = 0; x < cubeRoot1; x++) {
                if (index < numParticles1) {
                    double dx = spacing * (x - cubeRoot1 / 2.0);
                    double dy = spacing * (y - cubeRoot1 / 2.0);
                    double dz = spacing * (z - cubeRoot1 / 2.0);
                    if (dx * dx + dy * dy + dz * dz <= (cubeRoot1 / 2.0) * spacing * (cubeRoot1 / 2.0) * spacing) {
                        // Check if the point is within the sphere
                        particles.x[index] = spacing * x;
                        particles.y[index] = spacing * y;
                        particles.z[index] = spacing * z;
                        particles.vx[index] = 0.0;
                        particles.vy[index] = 0.0;
                        particles.vz[index] = 0.0;
                        particles.mass[index] = 1000.0;  // 假设所有粒子的质量为1

                        // 静态物理量初始化
                        particles.p[index] = 101325;  // 假定初始压力为大气压
                        particles.rho[index] = 1000;  // 假定初始密度，例如水的密度
                        particles.e[index] = 0;  // 初始能量设为0
                        particles.c[index] = 343;  // 假定初始声速，例如空气中的声速
                        particles.rho_prev[index] = 0;  // 上一时刻的密度
                        particles.e_prev[index] = 0;  // 上一时刻的能量
                        particles.h[index] = 0.05;  // 光滑核半径

                        // 动态物理量初始化
                        particles.drhodt[index] = 0;  // 密度变化率初始为0
                        particles.dedt[index] = 0;  // 能量变化率初始为0
                        particles.av_x[index] = 0;  // 平均速度初始化为0
                        particles.av_y[index] = 0;
                        particles.av_z[index] = 0;

                        // 张量初始化
                        for (int d = 0; d < 9; d++) {
                            particles.epsilon[index * 9 + d] = 0;
                            particles.tau[index * 9 + d] = 0;
                            particles.sigma[index * 9 + d] = 0;
                            particles.tau_prev[index * 9 + d] = 0;
                        }

                        // 其他初始化
                        particles.virt[index] = 0;  // 假定所有粒子最初都不是虚拟粒子

                        index++;
                    }
                    }
                }
            }
        }
        particles.mass[42] = 10000000000000.0;  // Special case for the 42nd particle

        for (int z = 0; z < cubeRoot2; z++) {
            for (int y = 0; y < cubeRoot2; y++) {
                for (int x = 0; x < cubeRoot2; x++) {
                    if (index < numParticles2) {
                        double dx = spacing * (x - cubeRoot2 / 2.0);
                        double dy = spacing * (y - cubeRoot2 / 2.0);
                        double dz = spacing * (z - cubeRoot2 / 2.0);
                        if (dx * dx + dy * dy + dz * dz <= (cubeRoot2 / 2.0) * spacing * (cubeRoot2 / 2.0) * spacing) {
                            // Check if the point is within the sphere
                            particles.x[index] = spacing * x;
                            particles.y[index] = 5 + spacing * y;
                            particles.z[index] = spacing * z;
                            particles.vx[index] = 0.0;
                            particles.vy[index] = 0.25;
                            particles.vz[index] = 0.0;
                            particles.mass[index] = 1000.0;  // 假设所有粒子的质量为1

                            // 静态物理量初始化
                            particles.p[index] = 101325;  // 假定初始压力为大气压
                            particles.rho[index] = 1000;  // 假定初始密度，例如水的密度
                            particles.e[index] = 0;  // 初始能量设为0
                            particles.c[index] = 343;  // 假定初始声速，例如空气中的声速
                            particles.rho_prev[index] = 0;  // 上一时刻的密度
                            particles.e_prev[index] = 0;  // 上一时刻的能量
                            particles.h[index] = 0.05;  // 光滑核半径

                            // 动态物理量初始化
                            particles.drhodt[index] = 0;  // 密度变化率初始为0
                            particles.dedt[index] = 0;  // 能量变化率初始为0
                            particles.av_x[index] = 0;  // 平均速度初始化为0
                            particles.av_y[index] = 0;
                            particles.av_z[index] = 0;

                            // 张量初始化
                            for (int d = 0; d < 9; d++) {
                                particles.epsilon[index * 9 + d] = 0;
                                particles.tau[index * 9 + d] = 0;
                                particles.sigma[index * 9 + d] = 0;
                                particles.tau_prev[index * 9 + d] = 0;
                            }

                            // 其他初始化
                            particles.virt[index] = 0;  // 假定所有粒子最初都不是虚拟粒子

                            index++;
                        }
                    }
                }
            }
        }




    }



void initialize_particles_cube(int numParticles, int cubeRoot, double spacing, Particles& particles) {
    int index = 0;
    for (int z = 0; z < cubeRoot; z++) {
        for (int y = 0; y < cubeRoot; y++) {
            for (int x = 0; x < cubeRoot; x++) {
                if (index < numParticles) {
                    // 位置和速度初始化
                    particles.x[index] = spacing * x;
                    particles.y[index] = spacing * y;
                    particles.z[index] = spacing * z;
                    particles.vx[index] = 0.0;
                    particles.vy[index] = 0.0;
                    particles.vz[index] = 0.0;
                    particles.mass[index] = 1000.0;  // 假设所有粒子的质量为1

                    // 静态物理量初始化
                    particles.p[index] = 101325;  // 假定初始压力为大气压
                    particles.rho[index] = 1000;  // 假定初始密度，例如水的密度
                    particles.e[index] = 0;  // 初始能量设为0
                    particles.c[index] = 343;  // 假定初始声速，例如空气中的声速
                    particles.rho_prev[index] = 0;  // 上一时刻的密度
                    particles.e_prev[index] = 0;  // 上一时刻的能量
                    particles.h[index] = 0.05;  // 光滑核半径

                    // 动态物理量初始化
                    particles.drhodt[index] = 0;  // 密度变化率初始为0
                    particles.dedt[index] = 0;  // 能量变化率初始为0
                    particles.av_x[index] = 0;  // 平均速度初始化为0
                    particles.av_y[index] = 0;
                    particles.av_z[index] = 0;

                    // 张量初始化
                    for (int d = 0; d < 9; d++) {
                        particles.epsilon[index * 9 + d] = 0;
                        particles.tau[index * 9 + d] = 0;
                        particles.sigma[index * 9 + d] = 0;
                        particles.tau_prev[index * 9 + d] = 0;
                    }

                    // 其他初始化
                    particles.virt[index] = 0;  // 假定所有粒子最初都不是虚拟粒子

                    index++;
                }
            }
        }
    }
    particles.mass[42] = 10000000000000.0;
}
