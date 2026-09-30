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
import copy
import json
import os
from pathlib import Path
import struct
import tempfile
import time
import unittest
import numpy as np
from final_polish import Expression as E,polish

class FinalistsTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.replay=Path(os.environ['SECANT_REPLAY_TEST_BIN']).resolve()
        self.x=np.linspace(-2,2,64,dtype=np.float32);self.y=2.75*self.x-.125
        self.model=E('add',(E('mul',(E.bank(0),E.input(0))),E.bank(1)))
        self.job=dict(seed=860,num_inputs=1,num_train_rows=64,num_validation_rows=64,
                      train_variance=float(np.var(self.y.astype(float))),test_variance=float(np.var(self.y.astype(float))))
        error=float(np.mean((self.x.astype(float)-self.y)**2))
        native=dict(genotype_hex=self.model.encode().hex(),resolved_ast_hex=self.model.resolve([1,0],0).encode().hex(),
                    coefficients=[1,0],permutation=0,fitted_leaves=[],train_nmse=error/self.job['train_variance'],solved=False)
        self.record=dict(native_result=native,train_mse=error,validation_mse=error,validation_r2=0,
                         finalists=dict(models=[]),best_expression='x0',accuracy_solution=0)

    def dataset(self,offset=0):
        path=self.root/('data'+str(offset))
        path.write_bytes(struct.pack('<8sIIQQ',b'SECSRDS\0',1,1,64,64)+
                         self.x.tobytes()+self.y.tobytes()+self.x.tobytes()+(self.y+np.float32(offset)).tobytes())
        return path

    def test_training_only_selection_and_provenance(self):
        before=copy.deepcopy(self.record)
        a=polish(self.record,self.job,self.dataset(),self.replay,time.monotonic()+5)
        b=polish(self.record,self.job,self.dataset(100),self.replay,time.monotonic()+5)
        self.assertTrue(a['final_polish']['accepted']);self.assertTrue(b['final_polish']['accepted'])
        self.assertEqual(a['resolved_ast_hex'],b['resolved_ast_hex'])
        self.assertGreater(a['validation_r2'],.999);self.assertLess(b['validation_r2'],0)
        self.assertNotIn('native_result',a)
        self.assertEqual(a['final_model']['resolved_ast_hex'],a['resolved_ast_hex'])
        self.assertEqual(a['search_result'],before)
        self.assertEqual(self.record,before)

    def test_expired_budget_preserves_incumbent(self):
        a=polish(self.record,self.job,self.root/'absent',self.replay,time.monotonic()-1)
        self.assertEqual(a['final_polish']['skipped'],'time_budget')
        self.assertEqual(a['train_mse'],self.record['train_mse'])

    def test_bad_replay_rejected(self):
        self.record['native_result']['genotype_hex']=E.constant(3).encode().hex()
        with self.assertRaisesRegex(ValueError,'replay failed'):
            polish(self.record,self.job,self.dataset(),self.replay,time.monotonic()+5)

if __name__=='__main__':unittest.main()
