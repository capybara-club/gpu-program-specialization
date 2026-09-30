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
from __future__ import annotations

from array import array
import ctypes
from dataclasses import dataclass
from pathlib import Path
import platform
import time
from typing import Sequence

from .genome import SystemGenome
from .packed_cubin import PackedModulePlan
from .shape import KernelShape


SSID_OK = 0
SSID_QUEUE_FULL = 5
SSID_TEMPLATE_ABI_VERSION = 3
SSID_OUTPUT_FULL_MSE = 0
SSID_OUTPUT_GENOME_WINNERS = 1
SSID_SETTINGS_MATERIALIZED = 0
SSID_SETTINGS_HASHED_INCUMBENT = 1


class C99RuntimeError(RuntimeError):
    pass


class CADTapeStats(ctypes.Structure):
    _fields_ = [
        ("node_count", ctypes.c_uint32),
        ("maximum_stack_depth", ctypes.c_uint32),
        ("scratch_floats_per_thread", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("tape_byte_count", ctypes.c_size_t),
    ]


class CAstDesc(ctypes.Structure):
    _fields_ = [
        ("byte_offset", ctypes.c_uint32),
        ("byte_count", ctypes.c_uint32),
        ("site_index", ctypes.c_uint16),
        ("reserved", ctypes.c_uint16),
    ]


class CGenomeDesc(ctypes.Structure):
    _fields_ = [
        ("first_ast", ctypes.c_uint32),
        ("ast_count", ctypes.c_uint16),
        ("reserved", ctypes.c_uint16),
    ]


class CGenomeBatch(ctypes.Structure):
    _fields_ = [
        ("genomes", ctypes.POINTER(CGenomeDesc)),
        ("genome_count", ctypes.c_uint32),
        ("asts", ctypes.POINTER(CAstDesc)),
        ("ast_count", ctypes.c_uint32),
        ("program_bytes", ctypes.POINTER(ctypes.c_uint8)),
        ("program_byte_count", ctypes.c_size_t),
    ]


class CPackedKernelDesc(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("genome_base", ctypes.c_uint32),
        ("genome_capacity", ctypes.c_uint32),
        ("genomes_per_cta", ctypes.c_uint32),
        ("register_count", ctypes.c_uint32),
        ("input_count", ctypes.c_uint32),
        ("output_count", ctypes.c_uint32),
        ("function_file_offset", ctypes.c_uint64),
        ("entry_file_offset", ctypes.c_uint64),
        ("arena_start_file_offset", ctypes.c_uint64),
        ("arena_instruction_count", ctypes.c_uint32),
        ("incoming_wait_mask", ctypes.c_uint32),
        ("continuation_file_offset", ctypes.c_uint64),
        ("dispatch_file_offsets", ctypes.POINTER(ctypes.c_uint64)),
        ("dispatch_instruction_words", ctypes.POINTER(ctypes.c_uint64)),
        ("dispatch_instruction_count", ctypes.c_uint32),
        ("cleanup_file_offsets", ctypes.POINTER(ctypes.c_uint64)),
        ("cleanup_count", ctypes.c_uint32),
        ("target_table_file_offsets", ctypes.POINTER(ctypes.c_uint64)),
        ("target_table_count", ctypes.c_uint32),
        ("register_count_file_offsets", ctypes.POINTER(ctypes.c_uint64)),
        ("register_count_offset_count", ctypes.c_uint32),
        ("register_count_header_file_offsets", ctypes.POINTER(ctypes.c_uint64)),
        ("register_count_header_offset_count", ctypes.c_uint32),
        ("input_registers", ctypes.POINTER(ctypes.c_uint8)),
        ("output_registers", ctypes.POINTER(ctypes.c_uint8)),
        ("available_registers", ctypes.POINTER(ctypes.c_uint8)),
        ("available_register_count", ctypes.c_uint32),
    ]


class CTemplateDesc(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("architecture", ctypes.c_uint32),
        ("cubin", ctypes.POINTER(ctypes.c_uint8)),
        ("cubin_byte_count", ctypes.c_size_t),
        ("kernels", ctypes.POINTER(CPackedKernelDesc)),
        ("kernel_count", ctypes.c_uint32),
        ("site_input_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("site_input_counts", ctypes.POINTER(ctypes.c_uint32)),
        ("site_count", ctypes.c_uint32),
        ("settings_mode", ctypes.c_uint32),
    ]


class CSpecializationStats(ctypes.Structure):
    _fields_ = [
        ("genome_count", ctypes.c_uint32),
        ("kernel_count", ctypes.c_uint32),
        ("populated_kernel_count", ctypes.c_uint32),
        ("maximum_register_count", ctypes.c_uint32),
        ("sass_instruction_count", ctypes.c_uint64),
    ]


class CRawBenchmarkResult(ctypes.Structure):
    _fields_ = [
        ("thread_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("module_count", ctypes.c_uint64),
        ("genome_count", ctypes.c_uint64),
        ("ast_count", ctypes.c_uint64),
        ("cubin_bytes_copied", ctypes.c_uint64),
        ("wall_seconds", ctypes.c_double),
        ("modules_per_second", ctypes.c_double),
        ("genomes_per_second", ctypes.c_double),
        ("asts_per_second", ctypes.c_double),
        ("cubin_gib_per_second", ctypes.c_double),
    ]


class CPipelineConfig(ctypes.Structure):
    _fields_ = [
        ("specialization_threads", ctypes.c_uint32),
        ("queue_capacity", ctypes.c_uint32),
        ("maximum_loaded_modules", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("enable_cuda", ctypes.c_uint32),
        ("execution_streams", ctypes.c_uint32),
    ]


class CFedbatchLaunch(ctypes.Structure):
    _fields_ = [
        ("settings_device", ctypes.c_uint64),
        ("settings_leading_dimension", ctypes.c_uint64),
        ("bindings_device", ctypes.c_uint64),
        ("bindings_leading_dimension", ctypes.c_uint64),
        ("num_settings", ctypes.c_uint64),
        ("num_genomes", ctypes.c_uint32),
        ("reference_device", ctypes.c_uint64),
        ("steps_per_observation", ctypes.c_uint32),
        ("mse_device", ctypes.c_uint64),
        ("threads_per_block", ctypes.c_uint32),
        ("shared_memory_bytes", ctypes.c_uint32),
        ("output_mode", ctypes.c_uint32),
        ("reduction_threads", ctypes.c_uint32),
        ("cta_score_device", ctypes.c_uint64),
        ("cta_setting_device", ctypes.c_uint64),
        ("winner_score_device", ctypes.c_uint64),
        ("winner_setting_device", ctypes.c_uint64),
    ]


class CTrajectoryLMLaunch(ctypes.Structure):
    _fields_ = [
        ("starts_device", ctypes.c_uint64),
        ("starts_leading_dimension", ctypes.c_uint64),
        ("bindings_device", ctypes.c_uint64),
        ("bindings_leading_dimension", ctypes.c_uint64),
        ("num_settings", ctypes.c_uint64),
        ("starts_per_setting", ctypes.c_uint32),
        ("reference_device", ctypes.c_uint64),
        ("steps_per_observation", ctypes.c_uint32),
        ("max_lm_iterations", ctypes.c_uint32),
        ("max_damping_attempts", ctypes.c_uint32),
        ("initial_damping", ctypes.c_float),
        ("constants_device", ctypes.c_uint64),
        ("constants_leading_dimension", ctypes.c_uint64),
        ("mse_device", ctypes.c_uint64),
        ("iterations_device", ctypes.c_uint64),
        ("accepted_steps_device", ctypes.c_uint64),
        ("threads_per_block", ctypes.c_uint32),
    ]


class CGPConfig(ctypes.Structure):
    _fields_ = [
        ("population_size", ctypes.c_uint32),
        ("settings_per_genome", ctypes.c_uint32),
        ("state_count", ctypes.c_uint32),
        ("constant_count", ctypes.c_uint32),
        ("max_nodes_per_ast", ctypes.c_uint32),
        ("max_program_bytes_per_ast", ctypes.c_uint32),
        ("initial_max_nodes", ctypes.c_uint32),
        ("max_depth", ctypes.c_uint32),
        ("tournament_size", ctypes.c_uint32),
        ("elite_count", ctypes.c_uint32),
        ("seed", ctypes.c_uint64),
        ("subtree_crossover_probability", ctypes.c_float),
        ("whole_site_crossover_probability", ctypes.c_float),
        ("subtree_mutation_probability", ctypes.c_float),
        ("point_mutation_probability", ctypes.c_float),
        ("binding_keep_probability", ctypes.c_float),
        ("constant_mutation_scale", ctypes.c_float),
        ("parsimony_coefficient", ctypes.c_float),
        ("minimum_valid_mse", ctypes.c_float),
    ]


class CGPGrammar(ctypes.Structure):
    _fields_ = [
        ("unary_operations", ctypes.POINTER(ctypes.c_uint8)),
        ("unary_operation_count", ctypes.c_uint32),
        ("binary_operations", ctypes.POINTER(ctypes.c_uint8)),
        ("binary_operation_count", ctypes.c_uint32),
    ]


class CGPRunConfig(ctypes.Structure):
    _fields_ = [
        ("generations", ctypes.c_uint32),
        ("steps_per_observation", ctypes.c_uint32),
        ("threads_per_block", ctypes.c_uint32),
        ("reduction_threads", ctypes.c_uint32),
        ("shared_memory_bytes", ctypes.c_uint32),
        ("output_mode", ctypes.c_uint32),
        ("reference_device", ctypes.c_uint64),
    ]


class CGPRunStats(ctypes.Structure):
    _fields_ = [
        ("generations", ctypes.c_uint32),
        ("batches_per_generation", ctypes.c_uint32),
        ("evaluated_genomes", ctypes.c_uint64),
        ("evaluated_configurations", ctypes.c_uint64),
        ("wall_seconds", ctypes.c_double),
        ("winner_materialization_seconds", ctypes.c_double),
        ("upload_seconds", ctypes.c_double),
        ("pipeline_seconds", ctypes.c_double),
        ("download_seconds", ctypes.c_double),
        ("evolution_seconds", ctypes.c_double),
        ("genomes_per_second", ctypes.c_double),
        ("configurations_per_second", ctypes.c_double),
        ("best_mse", ctypes.c_float),
        ("best_objective", ctypes.c_float),
        ("best_generation", ctypes.c_uint32),
        ("best_complexity", ctypes.c_uint32),
    ]


class CGPCheckpoint(ctypes.Structure):
    _fields_ = [
        ("elapsed_seconds", ctypes.c_double),
        ("generation_best_mse", ctypes.c_float),
        ("generation_best_objective", ctypes.c_float),
        ("best_mse", ctypes.c_float),
        ("best_objective", ctypes.c_float),
        ("generation", ctypes.c_uint32),
        ("best_generation", ctypes.c_uint32),
        ("best_complexity", ctypes.c_uint32),
    ]


class CGPTrace(ctypes.Structure):
    _fields_ = [
        ("checkpoints", ctypes.POINTER(CGPCheckpoint)),
        ("program_bytes", ctypes.POINTER(ctypes.c_uint8)),
        ("ast_byte_counts", ctypes.POINTER(ctypes.c_uint32)),
        ("constants", ctypes.POINTER(ctypes.c_float)),
        ("bindings", ctypes.POINTER(ctypes.c_uint32)),
        ("program_byte_count", ctypes.c_size_t),
        ("ast_byte_count_count", ctypes.c_size_t),
        ("constant_count", ctypes.c_size_t),
        ("binding_count", ctypes.c_size_t),
        ("checkpoint_capacity", ctypes.c_uint32),
        ("checkpoint_count", ctypes.c_uint32),
        ("checkpoint_stride", ctypes.c_uint32),
    ]


class CGPBest(ctypes.Structure):
    _fields_ = [
        ("mse", ctypes.c_float),
        ("objective", ctypes.c_float),
        ("generation", ctypes.c_uint32),
        ("complexity", ctypes.c_uint32),
        ("genome", CGenomeBatch),
        ("constants", ctypes.POINTER(ctypes.c_float)),
        ("bindings", ctypes.POINTER(ctypes.c_uint32)),
    ]


class CTicketResult(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("worker_index", ctypes.c_uint32),
        ("launched_kernel_count", ctypes.c_uint32),
        ("maximum_register_count", ctypes.c_uint32),
        ("sass_instruction_count", ctypes.c_uint64),
        ("queue_wait_seconds", ctypes.c_double),
        ("specialization_seconds", ctypes.c_double),
        ("module_load_seconds", ctypes.c_double),
        ("function_lookup_seconds", ctypes.c_double),
        ("launch_seconds", ctypes.c_double),
        ("total_seconds", ctypes.c_double),
    ]


@dataclass
class FlatGenomeBatch:
    """Contiguous Python-owned buffers borrowed by the C99 runner."""

    genome_descriptors: object
    ast_descriptors: object
    byte_buffer: object
    c_value: CGenomeBatch

    @classmethod
    def from_raw(
        cls,
        genomes: Sequence[tuple[int, int]],
        asts: Sequence[tuple[int, int, int]],
        program_bytes: bytes | bytearray | memoryview,
    ) -> "FlatGenomeBatch":
        if not genomes or not asts or not program_bytes:
            raise ValueError("flat genome batches cannot be empty")
        genome_array = (CGenomeDesc * len(genomes))(
            *(CGenomeDesc(first_ast, ast_count, 0) for first_ast, ast_count in genomes)
        )
        ast_array = (CAstDesc * len(asts))(
            *(CAstDesc(offset, count, site, 0) for offset, count, site in asts)
        )
        byte_values = bytes(program_bytes)
        byte_array = (ctypes.c_uint8 * len(byte_values)).from_buffer_copy(byte_values)
        c_value = CGenomeBatch(
            genome_array,
            len(genomes),
            ast_array,
            len(asts),
            byte_array,
            len(byte_values),
        )
        return cls(genome_array, ast_array, byte_array, c_value)

    @classmethod
    def from_genomes(
        cls,
        genomes: Sequence[SystemGenome],
        shape: KernelShape,
    ) -> "FlatGenomeBatch":
        genome_records: list[tuple[int, int]] = []
        ast_records: list[tuple[int, int, int]] = []
        program_bytes = bytearray()
        for genome in genomes:
            genome.validate(shape)
            first_ast = len(ast_records)
            for site_index, program in enumerate(genome.programs):
                offset = len(program_bytes)
                program_bytes.extend(program.data)
                ast_records.append((offset, len(program.data), site_index))
            genome_records.append((first_ast, len(genome.programs)))
        return cls.from_raw(genome_records, ast_records, program_bytes)

    @property
    def genome_count(self) -> int:
        return int(self.c_value.genome_count)

    @property
    def ast_count(self) -> int:
        return int(self.c_value.ast_count)

    @property
    def program_byte_count(self) -> int:
        return int(self.c_value.program_byte_count)


@dataclass(frozen=True)
class GPConfig:
    population_size: int = 128
    settings_per_genome: int = 512
    max_nodes_per_ast: int = 30
    max_program_bytes_per_ast: int = 192
    initial_max_nodes: int = 15
    max_depth: int = 12
    tournament_size: int = 5
    elite_count: int = 8
    seed: int = 7
    subtree_crossover_probability: float = 0.35
    whole_site_crossover_probability: float = 0.15
    subtree_mutation_probability: float = 0.25
    point_mutation_probability: float = 0.15
    binding_keep_probability: float = 0.5
    constant_mutation_scale: float = 0.5
    parsimony_coefficient: float = 0.0
    minimum_valid_mse: float = 0.0

    def c_value(self, shape: KernelShape) -> CGPConfig:
        return CGPConfig(
            self.population_size,
            self.settings_per_genome,
            shape.state_count,
            shape.constant_count,
            self.max_nodes_per_ast,
            self.max_program_bytes_per_ast,
            self.initial_max_nodes,
            self.max_depth,
            self.tournament_size,
            self.elite_count,
            self.seed,
            self.subtree_crossover_probability,
            self.whole_site_crossover_probability,
            self.subtree_mutation_probability,
            self.point_mutation_probability,
            self.binding_keep_probability,
            self.constant_mutation_scale,
            self.parsimony_coefficient,
            self.minimum_valid_mse,
        )


@dataclass(frozen=True)
class GPBest:
    mse: float
    objective: float
    generation: int
    complexity: int
    programs: tuple[bytes, ...]
    constants: tuple[float, ...]
    bindings: tuple[int, ...]


@dataclass(frozen=True)
class GPCandidate:
    index: int
    mse: float
    objective: float
    generation: int
    complexity: int
    programs: tuple[bytes, ...]
    constants: tuple[float, ...]
    bindings: tuple[int, ...]


@dataclass(frozen=True)
class GPCheckpoint:
    elapsed_seconds: float
    generation_best_mse: float
    generation_best_objective: float
    best_mse: float
    best_objective: float
    generation: int
    best_generation: int
    best_complexity: int
    programs: tuple[bytes, ...]
    constants: tuple[float, ...]
    bindings: tuple[int, ...]


class _TemplateDescriptorStorage:
    def __init__(self, cubin: bytes, plan: PackedModulePlan, shape: KernelShape, settings_mode: int):
        if plan.cubin_size != len(cubin):
            raise ValueError("CUBIN length does not match the packed module plan")
        self.keepalive: list[object] = []
        self.cubin = (ctypes.c_uint8 * len(cubin)).from_buffer_copy(cubin)
        self.keepalive.append(self.cubin)
        kernel_values: list[CPackedKernelDesc] = []
        for kernel in plan.kernels:
            site = kernel.cubin.site
            name = kernel.spec.name.encode("utf-8")
            dispatch_offsets = self._u64(site.dispatch_offsets)
            dispatch_words = self._u64(
                tuple(word for instruction in site.dispatch_instructions for word in instruction)
            )
            cleanup_offsets = self._u64(site.cleanup_offsets)
            target_offsets = self._u64(site.target_table_offsets)
            register_offsets = self._u64(kernel.cubin.register_count_offsets)
            register_header_offsets = self._u64(
                kernel.cubin.register_count_header_offsets
            )
            input_registers = self._u8(site.input_registers)
            output_registers = self._u8(site.output_registers)
            available_registers = self._u8(site.available_registers)
            self.keepalive.append(name)
            kernel_values.append(
                CPackedKernelDesc(
                    name,
                    kernel.spec.genome_base,
                    kernel.spec.dispatch.genome_capacity,
                    kernel.spec.dispatch.genomes_per_cta,
                    kernel.cubin.register_count,
                    kernel.cubin.input_count,
                    kernel.cubin.output_count,
                    kernel.cubin.function.file_offset,
                    site.entry_offset,
                    site.arena_start_offset,
                    site.arena_instruction_count,
                    site.incoming_wait_mask,
                    site.continuation_offset,
                    dispatch_offsets,
                    dispatch_words,
                    len(site.dispatch_offsets),
                    cleanup_offsets,
                    len(site.cleanup_offsets),
                    target_offsets,
                    len(site.target_table_offsets),
                    register_offsets,
                    len(kernel.cubin.register_count_offsets),
                    register_header_offsets,
                    len(kernel.cubin.register_count_header_offsets),
                    input_registers,
                    output_registers,
                    available_registers,
                    len(site.available_registers),
                )
            )
        self.kernels = (CPackedKernelDesc * len(kernel_values))(*kernel_values)
        self.site_offsets = self._u32(shape.ast_input_offsets)
        self.site_counts = self._u32(shape.ast_leaf_counts)
        self.keepalive.append(self.kernels)
        self.c_value = CTemplateDesc(
            SSID_TEMPLATE_ABI_VERSION,
            plan.architecture,
            self.cubin,
            len(cubin),
            self.kernels,
            len(kernel_values),
            self.site_offsets,
            self.site_counts,
            shape.ast_count,
            settings_mode,
        )

    def _u64(self, values: Sequence[int]):
        result = (ctypes.c_uint64 * len(values))(*values)
        self.keepalive.append(result)
        return result

    def _u32(self, values: Sequence[int]):
        result = (ctypes.c_uint32 * len(values))(*values)
        self.keepalive.append(result)
        return result

    def _u8(self, values: Sequence[int]):
        result = (ctypes.c_uint8 * len(values))(*values)
        self.keepalive.append(result)
        return result


def default_c99_library_path() -> Path:
    suffix = "dylib" if platform.system() == "Darwin" else "so"
    return Path(__file__).resolve().parents[2] / "build" / f"libsecant_system_id_c99.{suffix}"


class C99Library:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else default_c99_library_path()
        if not self.path.is_file():
            raise C99RuntimeError(f"C99 runtime is not built: run `make c99` in {self.path.parents[1]}")
        self.library = ctypes.CDLL(str(self.path))
        self._bind()

    def _bind(self) -> None:
        lib = self.library
        lib.ssid_last_error.argtypes = []
        lib.ssid_last_error.restype = ctypes.c_char_p
        lib.ssid_status_string.argtypes = [ctypes.c_int]
        lib.ssid_status_string.restype = ctypes.c_char_p
        lib.ssid_template_create.argtypes = [ctypes.POINTER(CTemplateDesc), ctypes.POINTER(ctypes.c_void_p)]
        lib.ssid_template_create.restype = ctypes.c_int
        lib.ssid_template_destroy.argtypes = [ctypes.c_void_p]
        lib.ssid_template_cubin_size.argtypes = [ctypes.c_void_p]
        lib.ssid_template_cubin_size.restype = ctypes.c_size_t
        lib.ssid_specialize_module.argtypes = [ctypes.c_void_p, ctypes.POINTER(CGenomeBatch), ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(CSpecializationStats)]
        lib.ssid_specialize_module.restype = ctypes.c_int
        lib.ssid_benchmark_specialization_raw.argtypes = [ctypes.c_void_p, ctypes.POINTER(CGenomeBatch), ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(CRawBenchmarkResult)]
        lib.ssid_benchmark_specialization_raw.restype = ctypes.c_int
        lib.ssid_ad_tape_build.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(CADTapeStats),
        ]
        lib.ssid_ad_tape_build.restype = ctypes.c_int
        lib.ssid_pipeline_create.argtypes = [ctypes.c_void_p, ctypes.POINTER(CPipelineConfig), ctypes.POINTER(ctypes.c_void_p)]
        lib.ssid_pipeline_create.restype = ctypes.c_int
        lib.ssid_module_pipeline_create.argtypes = [ctypes.c_size_t, ctypes.POINTER(CPipelineConfig), ctypes.POINTER(ctypes.c_void_p)]
        lib.ssid_module_pipeline_create.restype = ctypes.c_int
        lib.ssid_pipeline_submit.argtypes = [ctypes.c_void_p, ctypes.POINTER(CGenomeBatch), ctypes.POINTER(CFedbatchLaunch), ctypes.POINTER(ctypes.c_void_p)]
        lib.ssid_pipeline_submit.restype = ctypes.c_int
        lib.ssid_pipeline_submit_trajectory_lm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(CTrajectoryLMLaunch), ctypes.POINTER(ctypes.c_void_p)]
        lib.ssid_pipeline_submit_trajectory_lm.restype = ctypes.c_int
        lib.ssid_pipeline_wait_idle.argtypes = [ctypes.c_void_p]
        lib.ssid_pipeline_wait_idle.restype = ctypes.c_int
        lib.ssid_pipeline_destroy.argtypes = [ctypes.c_void_p]
        lib.ssid_ticket_poll.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(CTicketResult)]
        lib.ssid_ticket_poll.restype = ctypes.c_int
        lib.ssid_ticket_wait.argtypes = [ctypes.c_void_p, ctypes.POINTER(CTicketResult)]
        lib.ssid_ticket_wait.restype = ctypes.c_int
        lib.ssid_ticket_error.argtypes = [ctypes.c_void_p]
        lib.ssid_ticket_error.restype = ctypes.c_char_p
        lib.ssid_ticket_destroy.argtypes = [ctypes.c_void_p]
        lib.ssid_device_alloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
        lib.ssid_device_alloc.restype = ctypes.c_int
        lib.ssid_device_free.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.ssid_device_free.restype = ctypes.c_int
        lib.ssid_device_upload.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        lib.ssid_device_upload.restype = ctypes.c_int
        lib.ssid_device_download.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        lib.ssid_device_download.restype = ctypes.c_int
        lib.ssid_gp_create.argtypes = [ctypes.c_void_p, ctypes.POINTER(CGPConfig), ctypes.POINTER(CGPGrammar), ctypes.POINTER(ctypes.c_void_p)]
        lib.ssid_gp_create.restype = ctypes.c_int
        lib.ssid_gp_destroy.argtypes = [ctypes.c_void_p]
        lib.ssid_gp_generation_get.argtypes = [ctypes.c_void_p]
        lib.ssid_gp_generation_get.restype = ctypes.c_uint32
        lib.ssid_gp_batch_count.argtypes = [ctypes.c_void_p]
        lib.ssid_gp_batch_count.restype = ctypes.c_uint32
        lib.ssid_gp_batch_get.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(CGenomeBatch), ctypes.POINTER(ctypes.c_uint32)]
        lib.ssid_gp_batch_get.restype = ctypes.c_int
        lib.ssid_gp_settings_generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t]
        lib.ssid_gp_settings_generate.restype = ctypes.c_int
        lib.ssid_gp_winners_apply.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t]
        lib.ssid_gp_winners_apply.restype = ctypes.c_int
        lib.ssid_gp_generation_advance.argtypes = [ctypes.c_void_p]
        lib.ssid_gp_generation_advance.restype = ctypes.c_int
        lib.ssid_gp_best_get.argtypes = [ctypes.c_void_p, ctypes.POINTER(CGPBest)]
        lib.ssid_gp_best_get.restype = ctypes.c_int
        lib.ssid_gp_candidate_get.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(CGPBest)]
        lib.ssid_gp_candidate_get.restype = ctypes.c_int
        lib.ssid_gp_candidate_improve.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
        lib.ssid_gp_candidate_improve.restype = ctypes.c_int
        lib.ssid_gp_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(CGPRunConfig), ctypes.POINTER(CGPRunStats)]
        lib.ssid_gp_run.restype = ctypes.c_int
        lib.ssid_gp_run_traced.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(CGPRunConfig), ctypes.POINTER(CGPRunStats), ctypes.POINTER(CGPTrace)]
        lib.ssid_gp_run_traced.restype = ctypes.c_int

    def error(self, status: int, operation: str) -> C99RuntimeError:
        detail = self.library.ssid_last_error().decode("utf-8", "replace")
        name = self.library.ssid_status_string(status).decode("utf-8", "replace")
        return C99RuntimeError(f"{operation} failed: {name}: {detail}")

    def build_ad_tape(self, program: bytes, input_count: int) -> tuple[bytes, CADTapeStats]:
        if not program or input_count <= 0:
            raise ValueError("AD tape source and input count must be positive")
        source = (ctypes.c_uint8 * len(program)).from_buffer_copy(program)
        required = ctypes.c_size_t()
        stats = CADTapeStats()
        status = self.library.ssid_ad_tape_build(
            source, len(program), input_count, None, 0, ctypes.byref(required), ctypes.byref(stats)
        )
        if status != SSID_OK:
            raise self.error(status, "ssid_ad_tape_build(size)")
        output = (ctypes.c_uint8 * required.value)()
        status = self.library.ssid_ad_tape_build(
            source,
            len(program),
            input_count,
            output,
            required.value,
            ctypes.byref(required),
            ctypes.byref(stats),
        )
        if status != SSID_OK:
            raise self.error(status, "ssid_ad_tape_build")
        return bytes(output), stats

    def template(self, cubin: bytes, plan: PackedModulePlan, shape: KernelShape, settings_mode: int = SSID_SETTINGS_MATERIALIZED) -> "C99Template":
        storage = _TemplateDescriptorStorage(cubin, plan, shape, settings_mode)
        pointer = ctypes.c_void_p()
        status = self.library.ssid_template_create(ctypes.byref(storage.c_value), ctypes.byref(pointer))
        if status != SSID_OK:
            raise self.error(status, "ssid_template_create")
        return C99Template(self, pointer, len(cubin))

    def module_pipeline(
        self,
        cubin_byte_count: int,
        workers: int,
        queue_capacity: int,
        maximum_loaded_modules: int = 4,
        device_ordinal: int = 0,
        enable_cuda: bool = True,
        execution_streams: int = 1,
    ) -> "C99Pipeline":
        if cubin_byte_count <= 0:
            raise ValueError("module pipeline CUBIN size must be positive")
        config = CPipelineConfig(
            workers,
            queue_capacity,
            maximum_loaded_modules,
            device_ordinal,
            int(enable_cuda),
            execution_streams,
        )
        pointer = ctypes.c_void_p()
        status = self.library.ssid_module_pipeline_create(
            cubin_byte_count, ctypes.byref(config), ctypes.byref(pointer)
        )
        if status != SSID_OK:
            raise self.error(status, "ssid_module_pipeline_create")
        return C99Pipeline(self, None, pointer, enable_cuda)


class C99Template:
    def __init__(self, api: C99Library, pointer: ctypes.c_void_p, cubin_size: int):
        self.api = api
        self.pointer = pointer
        self.cubin_size = cubin_size

    def close(self) -> None:
        if self.pointer:
            self.api.library.ssid_template_destroy(self.pointer)
            self.pointer = ctypes.c_void_p()

    def __enter__(self) -> "C99Template":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def specialize(self, batch: FlatGenomeBatch) -> tuple[bytes, CSpecializationStats]:
        output = (ctypes.c_uint8 * self.cubin_size)()
        stats = CSpecializationStats()
        status = self.api.library.ssid_specialize_module(self.pointer, ctypes.byref(batch.c_value), output, self.cubin_size, ctypes.byref(stats))
        if status != SSID_OK:
            raise self.api.error(status, "ssid_specialize_module")
        return bytes(output), stats

    def benchmark_raw(self, batch: FlatGenomeBatch, threads: int, modules: int) -> CRawBenchmarkResult:
        result = CRawBenchmarkResult()
        status = self.api.library.ssid_benchmark_specialization_raw(self.pointer, ctypes.byref(batch.c_value), threads, modules, ctypes.byref(result))
        if status != SSID_OK:
            raise self.api.error(status, "ssid_benchmark_specialization_raw")
        return result

    def pipeline(
        self,
        workers: int,
        queue_capacity: int,
        maximum_loaded_modules: int = 4,
        device_ordinal: int = 0,
        enable_cuda: bool = False,
        execution_streams: int = 1,
    ) -> "C99Pipeline":
        config = CPipelineConfig(
            workers,
            queue_capacity,
            maximum_loaded_modules,
            device_ordinal,
            int(enable_cuda),
            execution_streams,
        )
        pointer = ctypes.c_void_p()
        status = self.api.library.ssid_pipeline_create(self.pointer, ctypes.byref(config), ctypes.byref(pointer))
        if status != SSID_OK:
            raise self.api.error(status, "ssid_pipeline_create")
        return C99Pipeline(self.api, self, pointer, enable_cuda)

    def gp(
        self,
        config: GPConfig,
        shape: KernelShape,
        unary_operations: Sequence[int],
        binary_operations: Sequence[int],
    ) -> "C99GP":
        unary = (ctypes.c_uint8 * len(unary_operations))(*unary_operations)
        binary = (ctypes.c_uint8 * len(binary_operations))(*binary_operations)
        grammar = CGPGrammar(unary, len(unary_operations), binary, len(binary_operations))
        c_config = config.c_value(shape)
        pointer = ctypes.c_void_p()
        status = self.api.library.ssid_gp_create(
            self.pointer, ctypes.byref(c_config), ctypes.byref(grammar), ctypes.byref(pointer)
        )
        if status != SSID_OK:
            raise self.api.error(status, "ssid_gp_create")
        return C99GP(self, pointer, config, shape)


class C99GP:
    def __init__(self, template: C99Template, pointer: ctypes.c_void_p, config: GPConfig, shape: KernelShape):
        self.template = template
        self.pointer = pointer
        self.config = config
        self.shape = shape

    def close(self) -> None:
        if self.pointer:
            self.template.api.library.ssid_gp_destroy(self.pointer)
            self.pointer = ctypes.c_void_p()

    def __enter__(self) -> "C99GP":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    @property
    def generation(self) -> int:
        return int(self.template.api.library.ssid_gp_generation_get(self.pointer))

    @property
    def batch_count(self) -> int:
        return int(self.template.api.library.ssid_gp_batch_count(self.pointer))

    def batch_programs(self, batch_index: int) -> tuple[tuple[bytes, ...], ...]:
        batch = CGenomeBatch()
        population_offset = ctypes.c_uint32()
        status = self.template.api.library.ssid_gp_batch_get(
            self.pointer, batch_index, ctypes.byref(batch), ctypes.byref(population_offset)
        )
        if status != SSID_OK:
            raise self.template.api.error(status, "ssid_gp_batch_get")
        genomes: list[tuple[bytes, ...]] = []
        for local_genome in range(int(batch.genome_count)):
            descriptor = batch.genomes[local_genome]
            programs: list[bytes] = []
            for local_ast in range(int(descriptor.ast_count)):
                ast = batch.asts[int(descriptor.first_ast) + local_ast]
                programs.append(bytes(batch.program_bytes[int(ast.byte_offset) : int(ast.byte_offset + ast.byte_count)]))
            genomes.append(tuple(programs))
        return tuple(genomes)

    def settings_generate(self) -> tuple[array, array]:
        settings = array("f", [0.0]) * (
            self.config.population_size * self.shape.constant_count * self.config.settings_per_genome
        )
        bindings = array("I", [0]) * (
            self.config.population_size * self.shape.input_count * self.config.settings_per_genome
        )
        settings_storage = (ctypes.c_float * len(settings)).from_buffer(settings)
        bindings_storage = (ctypes.c_uint32 * len(bindings)).from_buffer(bindings)
        status = self.template.api.library.ssid_gp_settings_generate(
            self.pointer, settings_storage, len(settings), bindings_storage, len(bindings)
        )
        if status != SSID_OK:
            raise self.template.api.error(status, "ssid_gp_settings_generate")
        return settings, bindings

    def winners_apply(
        self,
        winner_mse: array,
        winner_settings: array,
        settings: array,
        bindings: array,
    ) -> None:
        if len(winner_mse) != self.config.population_size or len(winner_settings) != self.config.population_size:
            raise ValueError("winner arrays must contain one entry per GP genome")
        score_storage = (ctypes.c_float * len(winner_mse)).from_buffer(winner_mse)
        winner_storage = (ctypes.c_uint32 * len(winner_settings)).from_buffer(winner_settings)
        settings_storage = (ctypes.c_float * len(settings)).from_buffer(settings)
        bindings_storage = (ctypes.c_uint32 * len(bindings)).from_buffer(bindings)
        status = self.template.api.library.ssid_gp_winners_apply(
            self.pointer,
            score_storage,
            winner_storage,
            settings_storage,
            len(settings),
            bindings_storage,
            len(bindings),
        )
        if status != SSID_OK:
            raise self.template.api.error(status, "ssid_gp_winners_apply")

    def advance(self) -> None:
        status = self.template.api.library.ssid_gp_generation_advance(self.pointer)
        if status != SSID_OK:
            raise self.template.api.error(status, "ssid_gp_generation_advance")

    def best(self) -> GPBest:
        value = CGPBest()
        status = self.template.api.library.ssid_gp_best_get(self.pointer, ctypes.byref(value))
        if status != SSID_OK:
            raise self.template.api.error(status, "ssid_gp_best_get")
        descriptor = value.genome.genomes[0]
        programs = []
        for local_ast in range(int(descriptor.ast_count)):
            ast = value.genome.asts[int(descriptor.first_ast) + local_ast]
            programs.append(bytes(value.genome.program_bytes[int(ast.byte_offset) : int(ast.byte_offset + ast.byte_count)]))
        constants = tuple(float(value.constants[index]) for index in range(self.shape.constant_count))
        bindings = tuple(int(value.bindings[index]) for index in range(self.shape.input_count))
        return GPBest(
            float(value.mse),
            float(value.objective),
            int(value.generation),
            int(value.complexity),
            tuple(programs),
            constants,
            bindings,
        )

    def candidate(self, index: int) -> GPCandidate:
        if not 0 <= index < self.config.population_size:
            raise IndexError("GP candidate index is outside the population")
        value = CGPBest()
        status = self.template.api.library.ssid_gp_candidate_get(
            self.pointer, index, ctypes.byref(value)
        )
        if status != SSID_OK:
            raise self.template.api.error(status, "ssid_gp_candidate_get")
        descriptor = value.genome.genomes[0]
        programs = []
        for local_ast in range(int(descriptor.ast_count)):
            ast = value.genome.asts[int(descriptor.first_ast) + local_ast]
            programs.append(
                bytes(
                    value.genome.program_bytes[
                        int(ast.byte_offset) : int(ast.byte_offset + ast.byte_count)
                    ]
                )
            )
        return GPCandidate(
            index,
            float(value.mse),
            float(value.objective),
            int(value.generation),
            int(value.complexity),
            tuple(programs),
            tuple(float(value.constants[offset]) for offset in range(self.shape.constant_count)),
            tuple(int(value.bindings[offset]) for offset in range(self.shape.input_count)),
        )

    def candidates(self) -> tuple[GPCandidate, ...]:
        return tuple(self.candidate(index) for index in range(self.config.population_size))

    def candidate_improve(
        self,
        index: int,
        mse: float,
        constants: Sequence[float],
        bindings: Sequence[int],
    ) -> bool:
        if len(constants) != self.shape.constant_count:
            raise ValueError("promoted constants do not match the GP shape")
        if len(bindings) != self.shape.input_count:
            raise ValueError("promoted bindings do not match the GP shape")
        constants_storage = (ctypes.c_float * len(constants))(*constants)
        bindings_storage = (ctypes.c_uint32 * len(bindings))(*bindings)
        accepted = ctypes.c_int()
        status = self.template.api.library.ssid_gp_candidate_improve(
            self.pointer,
            index,
            mse,
            constants_storage,
            len(constants),
            bindings_storage,
            len(bindings),
            ctypes.byref(accepted),
        )
        if status != SSID_OK:
            raise self.template.api.error(status, "ssid_gp_candidate_improve")
        return bool(accepted.value)

    def run(
        self,
        pipeline: "C99Pipeline",
        generations: int,
        reference_device: int,
        steps_per_observation: int,
        threads_per_block: int,
        reduction_threads: int = 256,
        full_mse_output: bool = False,
    ) -> CGPRunStats:
        config = CGPRunConfig(
            generations,
            steps_per_observation,
            threads_per_block,
            reduction_threads,
            self.shape.winner_shared_bytes(threads_per_block),
            SSID_OUTPUT_FULL_MSE if full_mse_output else SSID_OUTPUT_GENOME_WINNERS,
            reference_device,
        )
        stats = CGPRunStats()
        status = self.template.api.library.ssid_gp_run(
            self.pointer, pipeline.pointer, ctypes.byref(config), ctypes.byref(stats)
        )
        if status != SSID_OK:
            raise self.template.api.error(status, "ssid_gp_run")
        return stats

    def run_traced(
        self,
        pipeline: "C99Pipeline",
        generations: int,
        reference_device: int,
        steps_per_observation: int,
        threads_per_block: int,
        reduction_threads: int = 256,
        checkpoint_stride: int = 1,
        full_mse_output: bool = False,
    ) -> tuple[CGPRunStats, tuple[GPCheckpoint, ...]]:
        if generations <= 0:
            raise ValueError("generations must be positive")
        if checkpoint_stride <= 0:
            raise ValueError("checkpoint_stride must be positive")
        config = CGPRunConfig(
            generations,
            steps_per_observation,
            threads_per_block,
            reduction_threads,
            self.shape.winner_shared_bytes(threads_per_block),
            SSID_OUTPUT_FULL_MSE if full_mse_output else SSID_OUTPUT_GENOME_WINNERS,
            reference_device,
        )
        checkpoint_capacity = 1 + (generations - 1 + checkpoint_stride - 1) // checkpoint_stride
        checkpoint_storage = (CGPCheckpoint * checkpoint_capacity)()
        program_slot_bytes = self.config.max_program_bytes_per_ast
        program_storage = (ctypes.c_uint8 * (checkpoint_capacity * self.shape.ast_count * program_slot_bytes))()
        ast_count_storage = (ctypes.c_uint32 * (checkpoint_capacity * self.shape.ast_count))()
        constant_storage = (ctypes.c_float * (checkpoint_capacity * self.shape.constant_count))()
        binding_storage = (ctypes.c_uint32 * (checkpoint_capacity * self.shape.input_count))()
        trace = CGPTrace(
            checkpoint_storage,
            program_storage,
            ast_count_storage,
            constant_storage,
            binding_storage,
            len(program_storage),
            len(ast_count_storage),
            len(constant_storage),
            len(binding_storage),
            checkpoint_capacity,
            0,
            checkpoint_stride,
        )
        stats = CGPRunStats()
        status = self.template.api.library.ssid_gp_run_traced(
            self.pointer,
            pipeline.pointer,
            ctypes.byref(config),
            ctypes.byref(stats),
            ctypes.byref(trace),
        )
        if status != SSID_OK:
            raise self.template.api.error(status, "ssid_gp_run_traced")
        checkpoints: list[GPCheckpoint] = []
        for checkpoint_index in range(int(trace.checkpoint_count)):
            raw = checkpoint_storage[checkpoint_index]
            programs = []
            for site in range(self.shape.ast_count):
                byte_count = int(ast_count_storage[checkpoint_index * self.shape.ast_count + site])
                begin = (checkpoint_index * self.shape.ast_count + site) * program_slot_bytes
                programs.append(bytes(program_storage[begin : begin + byte_count]))
            constants = tuple(
                float(constant_storage[checkpoint_index * self.shape.constant_count + index])
                for index in range(self.shape.constant_count)
            )
            bindings = tuple(
                int(binding_storage[checkpoint_index * self.shape.input_count + index])
                for index in range(self.shape.input_count)
            )
            checkpoints.append(
                GPCheckpoint(
                    elapsed_seconds=float(raw.elapsed_seconds),
                    generation_best_mse=float(raw.generation_best_mse),
                    generation_best_objective=float(raw.generation_best_objective),
                    best_mse=float(raw.best_mse),
                    best_objective=float(raw.best_objective),
                    generation=int(raw.generation),
                    best_generation=int(raw.best_generation),
                    best_complexity=int(raw.best_complexity),
                    programs=tuple(programs),
                    constants=constants,
                    bindings=bindings,
                )
            )
        return stats, tuple(checkpoints)


class C99Ticket:
    def __init__(self, pipeline: "C99Pipeline", pointer: ctypes.c_void_p, keepalive: object):
        self.pipeline = pipeline
        self.pointer = pointer
        self.keepalive = keepalive

    def wait(self) -> CTicketResult:
        result = CTicketResult()
        status = self.pipeline.api.library.ssid_ticket_wait(self.pointer, ctypes.byref(result))
        if status != SSID_OK:
            detail = self.pipeline.api.library.ssid_ticket_error(self.pointer).decode("utf-8", "replace")
            raise C99RuntimeError(f"C99 pipeline ticket failed: {detail}")
        return result

    def close(self) -> None:
        if self.pointer:
            complete = ctypes.c_int()
            self.pipeline.api.library.ssid_ticket_poll(self.pointer, ctypes.byref(complete), None)
            if complete.value:
                self.pipeline.api.library.ssid_ticket_destroy(self.pointer)
                self.pointer = ctypes.c_void_p()
                self.keepalive = None


class C99Pipeline:
    def __init__(
        self,
        api: C99Library,
        template: C99Template | None,
        pointer: ctypes.c_void_p,
        cuda_enabled: bool,
    ):
        self.api = api
        self.template = template
        self.pointer = pointer
        self.cuda_enabled = cuda_enabled

    def close(self) -> None:
        if self.pointer:
            self.api.library.ssid_pipeline_destroy(self.pointer)
            self.pointer = ctypes.c_void_p()

    def __enter__(self) -> "C99Pipeline":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def submit(self, batch: FlatGenomeBatch, launch: CFedbatchLaunch | None = None) -> C99Ticket:
        if self.template is None:
            raise C99RuntimeError("raw module pipelines do not accept packed GP batches")
        pointer = ctypes.c_void_p()
        launch_pointer = None if launch is None else ctypes.byref(launch)
        status = self.api.library.ssid_pipeline_submit(self.pointer, ctypes.byref(batch.c_value), launch_pointer, ctypes.byref(pointer))
        if status != SSID_OK:
            raise self.api.error(status, "ssid_pipeline_submit")
        return C99Ticket(self, pointer, batch)

    def submit_trajectory_lm(self, cubin: bytes, launch: CTrajectoryLMLaunch) -> C99Ticket:
        if self.template is not None:
            raise C99RuntimeError("packed GP pipelines do not accept raw trajectory-LM modules")
        storage = (ctypes.c_uint8 * len(cubin)).from_buffer_copy(cubin)
        pointer = ctypes.c_void_p()
        status = self.api.library.ssid_pipeline_submit_trajectory_lm(
            self.pointer,
            storage,
            len(cubin),
            ctypes.byref(launch),
            ctypes.byref(pointer),
        )
        if status != SSID_OK:
            raise self.api.error(status, "ssid_pipeline_submit_trajectory_lm")
        return C99Ticket(self, pointer, storage)

    def allocate(self, byte_count: int) -> int:
        pointer = ctypes.c_uint64()
        status = self.api.library.ssid_device_alloc(self.pointer, byte_count, ctypes.byref(pointer))
        if status != SSID_OK:
            raise self.api.error(status, "ssid_device_alloc")
        return int(pointer.value)

    def free(self, pointer: int) -> None:
        status = self.api.library.ssid_device_free(self.pointer, pointer)
        if status != SSID_OK:
            raise self.api.error(status, "ssid_device_free")

    @staticmethod
    def _host_buffer(values: bytes | bytearray | array | memoryview):
        view = memoryview(values).cast("B")
        if view.readonly:
            storage = (ctypes.c_uint8 * len(view)).from_buffer_copy(view)
        else:
            storage = (ctypes.c_uint8 * len(view)).from_buffer(view)
        return storage, len(view)

    def upload(self, destination: int, values: bytes | bytearray | array | memoryview) -> None:
        storage, byte_count = self._host_buffer(values)
        status = self.api.library.ssid_device_upload(self.pointer, destination, storage, byte_count)
        if status != SSID_OK:
            raise self.api.error(status, "ssid_device_upload")

    def download(self, source: int, values: bytearray | array | memoryview) -> None:
        view = memoryview(values).cast("B")
        if view.readonly:
            raise ValueError("download destination must be writable")
        storage = (ctypes.c_uint8 * len(view)).from_buffer(view)
        status = self.api.library.ssid_device_download(self.pointer, storage, source, len(view))
        if status != SSID_OK:
            raise self.api.error(status, "ssid_device_download")


@dataclass(frozen=True)
class SpecializationBenchmark:
    workers: int
    submissions: int
    genome_count: int
    ast_count: int
    program_byte_count: int
    wall_seconds: float
    modules_per_second: float
    genomes_per_second: float
    asts_per_second: float
    mean_specialization_seconds: float
    maximum_specialization_seconds: float


def benchmark_specialization(template: C99Template, batch: FlatGenomeBatch, workers: int, submissions: int, queue_capacity: int | None = None) -> SpecializationBenchmark:
    if workers <= 0 or submissions <= 0:
        raise ValueError("workers and submissions must be positive")
    capacity = queue_capacity or submissions
    with template.pipeline(workers, capacity, enable_cuda=False) as pipeline:
        warmup = pipeline.submit(batch)
        warmup.wait()
        warmup.close()
        tickets: list[C99Ticket] = []
        started = time.perf_counter()
        for _ in range(submissions):
            tickets.append(pipeline.submit(batch))
        results = [ticket.wait() for ticket in tickets]
        elapsed = time.perf_counter() - started
        for ticket in tickets:
            ticket.close()
    specialization_times = [result.specialization_seconds for result in results]
    return SpecializationBenchmark(
        workers,
        submissions,
        batch.genome_count,
        batch.ast_count,
        batch.program_byte_count,
        elapsed,
        submissions / elapsed,
        submissions * batch.genome_count / elapsed,
        submissions * batch.ast_count / elapsed,
        sum(specialization_times) / len(specialization_times),
        max(specialization_times),
    )
