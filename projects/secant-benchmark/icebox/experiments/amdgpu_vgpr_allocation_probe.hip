#include <hip/hip_runtime.h>

extern "C" __global__
void secant_amdgpu_vgpr_allocation_probe(float* output) {
    float value = (float)threadIdx.x;
    asm volatile("v_mov_b32 v31, %0" : : "v"(value) : "v31");
    asm volatile("v_mov_b32 %0, v31" : "=v"(value));
    output[threadIdx.x] = value;
}
