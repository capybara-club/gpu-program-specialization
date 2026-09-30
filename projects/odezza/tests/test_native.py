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
"""Host-side ownership checks for failed native binding initialization."""
import ctypes as C
import unittest
from unittest.mock import patch

from odezza.native import LmPipeline, LmShape, NativeError


class Function:
    def __init__(self, call):
        self.call = call

    def __call__(self, *args):
        return self.call(*args)


class Library:
    def __init__(self, create_result=0, workspace_result=0, destroy_result=0):
        self.create_result = create_result
        self.workspace_result = workspace_result
        self.destroy_result = destroy_result
        self.destroy_calls = 0
        for name in ('create', 'workspace_requirements', 'destroy'):
            setattr(self, 'odezza_lm_pipeline_' + name, Function(getattr(self, name)))
        self.odezza_lm_pipeline_write_error = Function(lambda *args: 0)
        self.odezza_lm_pipeline_run = Function(lambda *args: 0)
        self.odezza_lm_pipeline_shape_report = Function(lambda *args: 0)
        self.odezza_lm_pipeline_create_with_fallback = Function(lambda info, mask, out: self.create(info, out))

    def create(self, info, out):
        C.cast(out, C.POINTER(C.c_void_p))[0] = 123
        return self.create_result

    def workspace_requirements(self, handle, size, alignment):
        C.cast(size, C.POINTER(C.c_size_t))[0] = 128
        C.cast(alignment, C.POINTER(C.c_size_t))[0] = 8
        return self.workspace_result

    def destroy(self, handle):
        self.destroy_calls += 1
        return self.destroy_result


class NativeOwnershipTests(unittest.TestCase):
    def create(self, library):
        with patch('odezza.native.C.CDLL', return_value=library):
            return LmPipeline(__file__, 120, LmShape(3, 3, 256, 1))

    def test_workspace_query_failure_releases_created_handle(self):
        library = Library(workspace_result=3)
        with self.assertRaises(NativeError):
            self.create(library)
        self.assertEqual(library.destroy_calls, 1)

    def test_fallback_rejects_unsupported_or_narrower_widths(self):
        for widths in ((0,), (3,), (1,), (True,)):
            with self.assertRaises(ValueError):
                LmPipeline(__file__, 120, LmShape(3,3,256,2), fallback_lanes=widths)

    def test_workspace_allocation_failure_releases_created_handle(self):
        library = Library()
        with patch('odezza.native.C.create_string_buffer', side_effect=MemoryError):
            with self.assertRaises(MemoryError):
                self.create(library)
        self.assertEqual(library.destroy_calls, 1)

    def test_failed_create_and_failed_cleanup_retain_handle_for_retry(self):
        library = Library(create_result=7, destroy_result=9)
        with self.assertRaises(NativeError) as caught:
            self.create(library)
        error = caught.exception
        self.assertEqual(error.result, 9)
        self.assertEqual(error.__cause__.result, 7)
        self.assertTrue(error.pipeline.handle)
        library.destroy_result = 0
        error.pipeline.close()
        self.assertFalse(error.pipeline.handle)
        self.assertEqual(library.destroy_calls, 2)


if __name__ == '__main__':
    unittest.main()
