#include <hip/hip_runtime.h>

#include <stddef.h>

extern "C" __global__
void
cusr_amd_island_probe(
    const float* input0,
    const float* input1,
    float* output,
    size_t count
) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float result;

    if (idx >= count) return;

    const float value0 = input0[idx];
    const float value1 = input1[idx];

    asm volatile(
        "s_nop 0x7fc0\n\t"
        "v_add_f32_e32 %0, 1.0, %1\n\t"
        "v_add_f32_e32 %0, 2.0, %2\n\t"
        ".rept 32\n\t"
        "s_nop 0\n\t"
        ".endr\n\t"
        "v_add_f32_e32 %0, 4.0, %1\n\t"
        "s_nop 0x7fc1\n\t"
        : "=&v"(result)
        : "v"(value0), "v"(value1)
        : "memory"
    );

    output[idx] = result;
}

extern "C" __global__
void
cusr_amd_instruction_donors(
    const float* input0,
    const float* input1,
    float* add_output,
    float* mul_output,
    size_t count
) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float add_result;
    float mul_result;

    if (idx >= count) return;

    const float value0 = input0[idx];
    const float value1 = input1[idx];

    asm volatile("v_add_f32_e32 %0, %1, %2" : "=v"(add_result) : "v"(value0), "v"(value1));
    asm volatile("v_mul_f32_e32 %0, %1, %2" : "=v"(mul_result) : "v"(value0), "v"(value1));

    add_output[idx] = add_result;
    mul_output[idx] = mul_result;
}
