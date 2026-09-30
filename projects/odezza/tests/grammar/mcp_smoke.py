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
"""A real stdio client, optionally over SSH; standard-library-only smoke test."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="rack1")
    parser.add_argument("--checkout", default="/home/cdurham/odezza/scratch/grammar_handoff_20260911")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    command = ["ssh", args.host, "env", "PYTHONPATH="+args.checkout+"/python",
               "ODEZZA_CORE_LIBRARY="+args.checkout+"/build/libodezza.so", "python3", "-m", "odezza.grammar",
               "--root", args.checkout+"/mcp-smoke", "mcp"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    counter = 0
    try:
        def rpc(method, params=None, notification=False):
            nonlocal counter
            counter += 1
            request = dict(jsonrpc="2.0", method=method)
            if params is not None: request["params"] = params
            if not notification: request["id"] = counter
            process.stdin.write(json.dumps(request)+"\n"); process.stdin.flush()
            if notification: return None
            response = json.loads(process.stdout.readline())
            assert response["id"] == counter and "error" not in response, response
            return response["result"]
        def tool(name, arguments):
            result = rpc("tools/call", dict(name=name, arguments=arguments))
            assert not result.get("isError"), result
            return result["structuredContent"]
        initialize = rpc("initialize", dict(protocolVersion="2025-06-18", capabilities={}, clientInfo={"name": "odezza-smoke", "version": "1"}))
        rpc("notifications/initialized", notification=True)
        tools = rpc("tools/list")["tools"]
        request = json.loads((Path(__file__).resolve().parents[2]/"examples/grammar/submit.json").read_text())
        start = time.monotonic()
        submitted = tool("odezza_submit", request)
        acknowledgement_seconds = time.monotonic()-start
        observed_statuses = []
        deadline = time.monotonic()+60
        while True:
            status = tool("odezza_status", {"job_id": submitted["job_id"]})
            observed_statuses.append(status["status"])
            if status["status"] not in ("queued", "preparing", "generating", "running"): break
            if time.monotonic() > deadline: raise TimeoutError("MCP smoke job exceeded 60 seconds")
            time.sleep(.1)
        report = tool("odezza_results", {"job_id": submitted["job_id"]})
        assert report["status"] == "complete", report
        candidate = report["winners"]["global"][0]
        assert candidate["values"] == [1.] and candidate["mse"] < 1e-10, candidate
        replay = tool("odezza_replay", dict(job_id=submitted["job_id"], candidate_id=candidate["id"], cpu=True))
        compact = tool("odezza_results", dict(job_id=submitted["job_id"], format="compact"))
        reference = compact["candidates"][compact["leaderboards"]["global"][0]]
        replay_index = tool("odezza_replay", dict(job_id=submitted["job_id"], address=reference["origin"], cpu=True))
        assert replay_index["values"] == replay["values"] and replay_index["resolved_programs"] == replay["resolved_programs"]
        assert replay_index["cpu_reference"] == replay["cpu_reference"]
        Path(args.output).write_text(json.dumps(dict(status="passed", initialize=initialize,
            tool_names=[t["name"] for t in tools], submitted=submitted, acknowledgement_seconds=acknowledgement_seconds,
            observed_statuses=observed_statuses, report=report, compact=compact, replay=replay, replay_index=replay_index), indent=2)+"\n")
        print(json.dumps(dict(status="passed", job_id=submitted["job_id"], mse=candidate["mse"], acknowledgement_seconds=acknowledgement_seconds)))
    finally:
        process.stdin.close()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.terminate(); process.wait(timeout=10)


if __name__ == "__main__": main()
