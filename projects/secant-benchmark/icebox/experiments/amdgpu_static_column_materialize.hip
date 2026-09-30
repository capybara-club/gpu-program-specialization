#include <stddef.h>
#include <stdint.h>

#include <hip/hip_runtime.h>

#define SECANT_AMDGPU_MATERIALIZE_PAD_INSTRUCTIONS 256u

template <uint32_t NumTraps>
static __device__ __forceinline__ void secant_amdgpu_materialize_emit_trap_pad(void) {
    #pragma unroll
    for (uint32_t instruction = 0u; instruction < NumTraps; ++instruction) {
        asm volatile("s_trap 2" ::: "memory");
    }
}

/*
 * Standalone AMDGPU ISA-inspection skeleton. It traps until the patch island
 * has been replaced.
 *
 * Input is column-major:
 *
 *     input[column * input_leading_dimension + row]
 *
 * Expected island order:
 *
 *     eight global loads
 *     S_TRAP load fence
 *     eight marker V_ADD_F32 instructions
 *     S_TRAP marker fence
 *     balanced seven-V_ADD_F32 reduction
 *     S_TRAP result fence and contiguous S_TRAP pad
 *     one global store consuming the reduction result
 */
extern "C" __global__
void secant_amdgpu_static_column_materialize_probe(
    const float* __restrict__ input,
    size_t input_leading_dimension,
    size_t num_rows,
    float* __restrict__ output
) {
    const size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) {
        return;
    }

    float input0 = input[(size_t)0u * input_leading_dimension + row];
    float input1 = input[(size_t)1u * input_leading_dimension + row];
    float input2 = input[(size_t)2u * input_leading_dimension + row];
    float input3 = input[(size_t)3u * input_leading_dimension + row];
    float input4 = input[(size_t)4u * input_leading_dimension + row];
    float input5 = input[(size_t)5u * input_leading_dimension + row];
    float input6 = input[(size_t)6u * input_leading_dimension + row];
    float input7 = input[(size_t)7u * input_leading_dimension + row];

    asm volatile(
        "s_trap 2"
        : "+v"(input0), "+v"(input1), "+v"(input2), "+v"(input3),
          "+v"(input4), "+v"(input5), "+v"(input6), "+v"(input7)
        :
        : "memory");

    float marked0;
    float marked1;
    float marked2;
    float marked3;
    float marked4;
    float marked5;
    float marked6;
    float marked7;
    asm volatile(
        "v_add_f32 %0, 0x7fc0ffee, %8\n\t"
        "v_add_f32 %1, 0x7fc0ffef, %9\n\t"
        "v_add_f32 %2, 0x7fc0fff0, %10\n\t"
        "v_add_f32 %3, 0x7fc0fff1, %11\n\t"
        "v_add_f32 %4, 0x7fc0fff2, %12\n\t"
        "v_add_f32 %5, 0x7fc0fff3, %13\n\t"
        "v_add_f32 %6, 0x7fc0fff4, %14\n\t"
        "v_add_f32 %7, 0x7fc0fff5, %15\n\t"
        "s_trap 2"
        : "=v"(marked0), "=v"(marked1), "=v"(marked2), "=v"(marked3),
          "=v"(marked4), "=v"(marked5), "=v"(marked6), "=v"(marked7)
        : "v"(input0), "v"(input1), "v"(input2), "v"(input3),
          "v"(input4), "v"(input5), "v"(input6), "v"(input7)
        : "memory");

    const float reduction01 = marked0 + marked1;
    const float reduction23 = marked2 + marked3;
    const float reduction45 = marked4 + marked5;
    const float reduction67 = marked6 + marked7;
    const float reduction0123 = reduction01 + reduction23;
    const float reduction4567 = reduction45 + reduction67;
    float result = reduction0123 + reduction4567;

    asm volatile("s_trap 2" : "+v"(result) : : "memory");
    secant_amdgpu_materialize_emit_trap_pad<SECANT_AMDGPU_MATERIALIZE_PAD_INSTRUCTIONS>();

    output[row] = result;
}
