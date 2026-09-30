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
"""mac1 client for the private trial. Standard library only; no AST expansion.

Uses the basic NATS request/reply wire protocol through an authenticated SSH
tunnel. Serialized calls per connection; create separate clients for concurrency.
No automatic submission retries or implicit recovery across service generations.
"""
import argparse
import hashlib
import json
import socket
import sys
import time
import uuid

PREFIX = 'odezza.trial.v1'
CHUNK = 256 * 1024


class Client:
    def __init__(self, host='127.0.0.1', port=14222, timeout=10):
        self.socket = socket.create_connection((host, port), timeout)
        self.reader = self.socket.makefile('rb')
        line = self.reader.readline(16384)
        if not line.startswith(b'INFO '):
            self.close()
            raise RuntimeError('expected NATS INFO')
        self.socket.sendall(b'CONNECT {"verbose":false,"pedantic":true,"lang":"python","version":"1","protocol":1}\r\n')

    def close(self):
        self.reader.close()
        self.socket.close()

    def rpc(self, operation, data=b''):
        inbox = '_INBOX.' + uuid.uuid4().hex
        subject = PREFIX + '.' + operation
        if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.' for c in subject):
            raise ValueError('invalid operation')
        if len(data) > CHUNK:
            raise ValueError('message limit')
        self.socket.sendall(f'SUB {inbox} 1\r\nUNSUB 1 1\r\nPUB {subject} {inbox} {len(data)}\r\n'.encode() + data + b'\r\n')
        while True:
            line = self.reader.readline(16384)
            if not line:
                raise ConnectionError('broker connection closed')
            if line == b'PING\r\n':
                self.socket.sendall(b'PONG\r\n')
                continue
            if line.startswith((b'INFO ', b'+OK', b'PONG')):
                continue
            if not line.startswith(b'MSG '):
                raise RuntimeError(line.decode(errors='replace'))
            fields = line.split()
            size = int(fields[-1])
            if not 0 <= size <= CHUNK + 8192:
                raise RuntimeError('reply limit')
            payload = self.reader.read(size)
            if len(payload) != size or self.reader.read(2) != b'\r\n':
                raise ConnectionError('truncated reply')
            if fields[1].decode() != inbox:
                raise RuntimeError('unexpected reply identity')
            if payload.startswith(b'ERR\n'):
                raise RuntimeError(payload[4:].decode())
            if not payload.startswith(b'OK\n'):
                raise RuntimeError('invalid reply')
            return payload[3:]

    def health(self):
        return json.loads(self.rpc('health'))

    def submit(self, raw, job_id=None, generation=None):
        if generation is None:
            generation = self.health()['generation']
        job_id = job_id or uuid.uuid4().hex
        handle = generation + '.' + job_id
        self.rpc(f'begin.{handle}.{len(raw)}')
        for start in range(0, len(raw), CHUNK):
            self.rpc(f'put.{handle}.{start}', raw[start:start + CHUNK])
        status = json.loads(self.rpc(f'submit.{handle}'))
        return dict(handle=handle, request_sha256=hashlib.sha256(raw).hexdigest(), **status)

    def status(self, handle):
        return json.loads(self.rpc('status.' + handle))

    def result(self, handle):
        status = self.status(handle)
        if status['state'] != 'terminal':
            raise RuntimeError('job_active')
        size = status['report_bytes']
        result = bytearray()
        while len(result) < size:
            chunk = self.rpc(f'result.{handle}.{len(result)}')
            if not chunk or len(result) + len(chunk) > size:
                raise RuntimeError('invalid report length')
            result.extend(chunk)
        return json.loads(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=14222)
    p.add_argument('command', choices=['health', 'jobs', 'submit', 'status', 'result', 'cancel', 'release', 'wait'])
    p.add_argument('value', nargs='?')
    p.add_argument('--timeout', type=float, default=3700)
    args = p.parse_args()
    client = Client(port=args.port)
    try:
        if args.command == 'health':
            result = client.health()
        elif args.command == 'jobs':
            result = json.loads(client.rpc('jobs'))
        elif args.command == 'submit':
            with open(args.value, 'rb') as f:
                raw = f.read(8 * 1024 * 1024 + 1)
            result = client.submit(raw)
        elif args.command == 'result':
            result = client.result(args.value)
        elif args.command == 'wait':
            deadline = time.monotonic() + args.timeout
            while True:
                status = client.status(args.value)
                if status['state'] == 'terminal':
                    result = dict(service=status, report=client.result(args.value))
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('client_wait_expired; job remains active')
                time.sleep(.25)
        else:
            raw = client.rpc(args.command + '.' + args.value)
            result = json.loads(raw) if raw else {'released': True}
        print(json.dumps(result, indent=2))
    finally:
        client.close()


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
