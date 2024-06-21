#include <cmath>
#include <cstdlib> // For rand()
#include "datatypes.cuh"

double random_double(double min, double max);
// Calculate the radius of a sphere based on the number of particles and their spacing
//double calculate_sphere_radius(int numParticlesPerSphere, double spacing);

// Initialize particles in two spheres

void initialize_particles_sphere(int numParticles1, int numParticles2, int cubeRoot1, int cubeRoot2, double spacing, Particles& particles);

void initialize_particles_cube(int numParticles, int cubeRoot, double spacing, Particles& particles);