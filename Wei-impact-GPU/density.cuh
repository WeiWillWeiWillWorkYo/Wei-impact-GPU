#include <iostream>
#include <cuda_runtime.h>
#include "datatypes.cuh"

void runNormSumDensity(Particles& parts, InteractionPairs& pairs);

void runSumDensity(Particles& parts, InteractionPairs& pairs);

void runConDensity(Particles& parts, InteractionPairs& pairs);

void runCopy2InitDensity(Particles& parts);