# SPDX-FileCopyrightText: 2026 Charles Durham
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2026 Charles Durham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# SPDX-FileCopyrightText: 2026 Charles Durham
#
# SPDX-License-Identifier: MIT

from mm_ptx.stack_ptx import (
    StackTypeEnum,
    ArgTypeEnum,
    create_instruction_enum,
    create_special_register_enum,
    StackPtx,
)

from enum import auto, unique

@unique
class Stack(StackTypeEnum):
    # Fields: (auto-id, ptx literal prefix)
    F32		= (auto(), "f32")
    U32		= (auto(), "u32")
    S32		= (auto(), "s32")
    PRED	= (auto(), "pred")
    TF32	= (auto(), "b32")
    F16		= (auto(), "b16")
    F16X2	= (auto(), "b32")
    U8		= (auto(), "u8")


@unique
class ArgType(ArgTypeEnum):
    # Fields: (auto-id, stack type enum, optional num_vec_elems)
    F32		= (auto(), Stack.F32)
    TF32	= (auto(), Stack.TF32)
    V1_TF32	= (auto(), Stack.TF32,  1)
    V2_TF32	= (auto(), Stack.TF32,  2)
    V4_F32	= (auto(), Stack.F32,   4)
    S32		= (auto(), Stack.S32)
    U32		= (auto(), Stack.U32)
    V4_U32	= (auto(), Stack.U32,   4)
    PRED	= (auto(), Stack.PRED)
    F16		= (auto(), Stack.F16)
    F16X2	= (auto(), Stack.F16X2)
    V2_F16	= (auto(), Stack.F16,   2)
    U8		= (auto(), Stack.U8)
    V4_U8	= (auto(), Stack.U8,    4)


@unique
class PtxInstruction(create_instruction_enum(ArgType)):
    # Fields: (auto-id, ptx opcode string, [arg types], [ret types], optional aligned flag)
    add_u32			        = (auto(), "add.u32",               [ArgType.U32, ArgType.U32],                 [ArgType.U32])
    sub_u32			        = (auto(), "sub.u32",               [ArgType.U32, ArgType.U32],                 [ArgType.U32])
    mul_lo_u32		        = (auto(), "mul.lo.u32",            [ArgType.U32, ArgType.U32],                 [ArgType.U32])
    mad_lo_u32		        = (auto(), "mad.lo.u32",            [ArgType.U32, ArgType.U32, ArgType.U32],    [ArgType.U32])

    copysign_f32	        = (auto(), "copysign.f32",          [ArgType.F32, ArgType.F32],                 [ArgType.F32])
    add_ftz_f32		        = (auto(), "add.ftz.f32",           [ArgType.F32, ArgType.F32],                 [ArgType.F32])
    sub_ftz_f32		        = (auto(), "sub.ftz.f32",           [ArgType.F32, ArgType.F32],                 [ArgType.F32])
    mul_ftz_f32		        = (auto(), "mul.ftz.f32",           [ArgType.F32, ArgType.F32],                 [ArgType.F32])
    fma_rn_ftz_f32	        = (auto(), "fma.rn.ftz.f32",        [ArgType.F32, ArgType.F32, ArgType.F32],    [ArgType.F32])
    div_approx_ftz_f32	    = (auto(), "div.approx.ftz.f32",    [ArgType.F32, ArgType.F32],                 [ArgType.F32])
    abs_ftz_f32		        = (auto(), "abs.ftz.f32",           [ArgType.F32],                              [ArgType.F32])
    neg_ftz_f32		        = (auto(), "neg.ftz.f32",           [ArgType.F32],                              [ArgType.F32])
    min_ftz_f32		        = (auto(), "min.ftz.f32",           [ArgType.F32, ArgType.F32],                 [ArgType.F32])
    max_ftz_f32		        = (auto(), "max.ftz.f32",           [ArgType.F32, ArgType.F32],                 [ArgType.F32])

    rcp_approx_ftz_f32	    = (auto(), "rcp.approx.ftz.f32",    [ArgType.F32],                              [ArgType.F32])
    sqrt_approx_ftz_f32	    = (auto(), "sqrt.approx.ftz.f32",   [ArgType.F32],                              [ArgType.F32])
    rsqrt_approx_ftz_f32    = (auto(), "rsqrt.approx.ftz.f32",  [ArgType.F32],                              [ArgType.F32])
    sin_approx_ftz_f32	    = (auto(), "sin.approx.ftz.f32",    [ArgType.F32],                              [ArgType.F32])
    cos_approx_ftz_f32	    = (auto(), "cos.approx.ftz.f32",    [ArgType.F32],                              [ArgType.F32])
    lg2_approx_ftz_f32	    = (auto(), "lg2.approx.ftz.f32",    [ArgType.F32],                              [ArgType.F32])
    ex2_approx_ftz_f32	    = (auto(), "ex2.approx.ftz.f32",    [ArgType.F32],                              [ArgType.F32])
    tanh_approx_f32	        = (auto(), "tanh.approx.f32",       [ArgType.F32],                              [ArgType.F32])

    setp_lt_ftz_f32	        = (auto(), "setp.lt.ftz.f32",       [ArgType.F32, ArgType.F32],                 [ArgType.PRED])
    setp_gt_ftz_f32	        = (auto(), "setp.gt.ftz.f32",       [ArgType.F32, ArgType.F32],                 [ArgType.PRED])
    setp_eq_u32		        = (auto(), "setp.eq.u32",           [ArgType.U32, ArgType.U32],                 [ArgType.PRED])
    setp_ne_u32		        = (auto(), "setp.ne.u32",           [ArgType.U32, ArgType.U32],                 [ArgType.PRED])
    selp_f32			    = (auto(), "selp.f32",              [ArgType.F32, ArgType.F32, ArgType.PRED],   [ArgType.F32])

    cvt_rna_tf32_f32	    = (auto(), "cvt.rna.tf32.f32",      [ArgType.F32],                              [ArgType.TF32])
    cvt_rn_f32_u32	        = (auto(), "cvt.rn.f32.u32",        [ArgType.U32],                              [ArgType.F32])

    f16x2_to_v2_f16	        = (auto(), "mov.b32",               [ArgType.F16X2],                            [ArgType.V2_F16])
    f16_to_f32		        = (auto(), "cvt.f32.f16",           [ArgType.F16],                              [ArgType.F32])

    u32_to_v4_u8		    = (auto(), "mov.b32",               [ArgType.U32],                              [ArgType.V4_U8])
    cvt_u32_u8		        = (auto(), "cvt.u32.u8",            [ArgType.U8],                               [ArgType.U32])

    mma_sync_aligned_m16n8k4_row_col_f32_tf32_tf32_f32	= (
        auto(),
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32",
        [ArgType.V2_TF32, ArgType.V1_TF32, ArgType.V4_F32],
        [ArgType.V4_F32],
        True,
    )


@unique
class SpecialRegister(create_special_register_enum(ArgType)):
    # Fields: (auto-id, ptx register name, arg type)
    tid			        = (auto(), "tid",               ArgType.V4_U32)
    tid_x		        = (auto(), "tid.x",             ArgType.U32)
    tid_y		        = (auto(), "tid.y",             ArgType.U32)
    tid_z		        = (auto(), "tid.z",             ArgType.U32)

    ntid		        = (auto(), "ntid",              ArgType.V4_U32)
    ntid_x		        = (auto(), "ntid.x",            ArgType.U32)
    ntid_y		        = (auto(), "ntid.y",            ArgType.U32)
    ntid_z		        = (auto(), "ntid.z",            ArgType.U32)

    laneid		        = (auto(), "laneid",            ArgType.U32)
    warpid		        = (auto(), "warpid",            ArgType.U32)
    nwarpid		        = (auto(), "nwarpid",           ArgType.U32)

    ctaid		        = (auto(), "ctaid",             ArgType.V4_U32)
    ctaid_x		        = (auto(), "ctaid.x",           ArgType.U32)
    ctaid_y		        = (auto(), "ctaid.y",           ArgType.U32)
    ctaid_z		        = (auto(), "ctaid.z",           ArgType.U32)

    nctaid		        = (auto(), "nctaid",            ArgType.V4_U32)
    nctaid_x            = (auto(), "nctaid.x",          ArgType.U32)
    nctaid_y	        = (auto(), "nctaid.y",          ArgType.U32)
    nctaid_z	        = (auto(), "nctaid.z",          ArgType.U32)

    smid		        = (auto(), "smid",              ArgType.U32)
    nsmid		        = (auto(), "nsmid",             ArgType.U32)
    gridid		        = (auto(), "gridid",            ArgType.U32)

    lanemask_eq	        = (auto(), "lanemask_eq",       ArgType.U32)
    lanemask_le	        = (auto(), "lanemask_le",       ArgType.U32)
    lanemask_lt	        = (auto(), "lanemask_lt",       ArgType.U32)
    lanemask_ge	        = (auto(), "lanemask_ge",       ArgType.U32)
    lanemask_gt	        = (auto(), "lanemask_gt",       ArgType.U32)

    clock		        = (auto(), "clock",             ArgType.U32)
    clock_hi            = (auto(), "clock_hi",          ArgType.U32)

    total_smem_size	    = (auto(), "total_smem_size",   ArgType.U32)
    aggr_smem_size	    = (auto(), "aggr_smem_size",    ArgType.U32)
    dynamic_smem_size	= (auto(), "dynamic_smem_size", ArgType.U32)


compiler = StackPtx(
    stack_enum=Stack,
    arg_enum=ArgType,
    ptx_instruction_enum=PtxInstruction,
    special_register_enum=SpecialRegister,
)
