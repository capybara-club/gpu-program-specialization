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
import io
import json
import unittest
from unittest.mock import patch
from odezza.grammar.mcp import Transport, serve
from odezza.grammar.native_service import NativeService, ServiceError


class FakeNative:
    supported_tools=NativeService.supported_tools
    def release(self, **kwargs):return dict(released=True,**kwargs)
    def submit(self, **kwargs):raise ServiceError('admission_limit','Queue is full')


class NativeTransportTests(unittest.TestCase):
    def ready(self):
        t=Transport(FakeNative())
        t.dispatch(dict(jsonrpc='2.0',id=1,method='initialize'))
        t.dispatch(dict(jsonrpc='2.0',method='notifications/initialized'))
        return t

    def call(self,t,name,args):
        return t.dispatch(dict(jsonrpc='2.0',id=2,method='tools/call',params=dict(name=name,arguments=args)))

    def test_native_schema(self):
        tools={t['name']:t for t in self.ready().tools}
        self.assertNotIn('plan_only',tools['odezza_submit']['inputSchema']['properties'])
        self.assertIn('odezza_release',tools)
        self.assertNotIn('odezza_fit',tools)

    def test_release_alternatives(self):
        t=self.ready()
        self.assertFalse(self.call(t,'odezza_release',dict(job_id='a'))['result']['isError'])
        self.assertIn('error',self.call(t,'odezza_release',dict(job_id='a',problem_id='b')))
        self.assertIn('error',self.call(t,'odezza_release',{}))

    def test_structured_error(self):
        result=self.call(self.ready(),'odezza_submit',dict(problem={},grammar={}))['result']
        self.assertTrue(result['isError'])
        self.assertEqual(result['structuredContent']['error']['code'],'admission_limit')

    def test_utf8_limit_and_recovery(self):
        # The transport limit is bytes, even when reading a text stream.
        source=io.StringIO('"'+'é'*40+'"\n'+json.dumps(dict(jsonrpc='2.0',id=7,method='ping'))+'\n')
        output=io.StringIO()
        with patch('odezza.grammar.mcp.MAX_MESSAGE_BYTES',64):serve(object(),source,output)
        results=[json.loads(s) for s in output.getvalue().splitlines()]
        self.assertEqual(results[0]['error']['data']['code'],'message_too_large')
        self.assertEqual(results[1]['id'],7)

    def test_device_cli_does_not_consume_subcommand(self):
        from contextlib import redirect_stdout
        from odezza.grammar.__main__ import main, device_list
        import argparse
        output=io.StringIO()
        with redirect_stdout(output):self.assertEqual(main(['--devices','0,1','capabilities']),0)
        result=json.loads(output.getvalue())
        self.assertEqual(result['selected_devices'],[0,1])
        self.assertTrue(result['process_supervision'])
        for value in ('0,0','-1','4294967296','0,',''):
            with self.assertRaises(argparse.ArgumentTypeError):device_list(value)
