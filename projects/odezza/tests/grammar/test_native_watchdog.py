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
import threading
import time
import unittest
from unittest.mock import Mock, patch
from odezza.grammar.native_supervisor import SupervisedNativeService


class WatchdogTests(unittest.TestCase):
    def test_healthy_heartbeat_does_not_hide_stalled_initializer(self):
        service=SupervisedNativeService.__new__(SupervisedNativeService)
        service.stop=Mock();service.stop.wait.side_effect=[False,True]
        service.lock=threading.RLock();service.closed=False
        service.options={};service.error_times={};service.terminal={};service.errors={}
        service.process=Mock();service.process.is_alive.return_value=True
        service.last_health=time.monotonic();service.watchdog_seconds=30
        service.max_worker_rss_bytes=service.max_total_rss_bytes=10**12
        service.health={'rss_bytes':1024,'statuses':{'a':{'status':'initializing','timing':{'worker_seconds':31}}}}
        service._lost=Mock()
        with patch('odezza.grammar.native_supervisor._rss_bytes',return_value=1024):service._monitor()
        service._lost.assert_called_once_with('CUDA initialization exceeded its watchdog deadline')
