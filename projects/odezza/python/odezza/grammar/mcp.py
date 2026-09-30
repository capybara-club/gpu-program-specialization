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
"""Small stdio MCP 2025-06-18 transport. No network listener or external SDK."""
from __future__ import annotations
import json
import sys
from copy import deepcopy

MAX_MESSAGE_BYTES = 16*1024*1024



def schema(required, **properties):
    return dict(type="object", properties=properties, required=required, additionalProperties=False)


STRING, OBJECT = {"type": "string"}, {"type": "object"}
SUBMISSION = schema(["grammar"], problem=OBJECT, problem_id=STRING, grammar=OBJECT, execution=OBJECT,
                    idempotency_key=STRING, plan_only={"type": "boolean"})
SUBMISSION["oneOf"] = [{"required": ["problem"]}, {"required": ["problem_id"]}]
REPLAY = schema(["job_id"], job_id=STRING, candidate_id=STRING, address=OBJECT, cpu={"type": "boolean"})
REPLAY["oneOf"] = [{"required": ["candidate_id"]}, {"required": ["address"]}]
RELEASE = schema([], job_id=STRING, problem_id=STRING)
RELEASE["oneOf"] = [{"required": ["job_id"]}, {"required": ["problem_id"]}]
TOOLS = [
    dict(name="odezza_release", description="Release completed job results and idempotency keys, or a prepared problem. Export results first. Active jobs must finish or be cancelled before release.", inputSchema=RELEASE),
    dict(name="odezza_capabilities", description="Inspect executable grammar, solver, observation and RNG capabilities.", inputSchema=schema([])),
    dict(name="odezza_prepare", description="Prepare ordered states, trajectories with explicit initial states, optional masks and known RHS.", inputSchema=schema(["problem"], problem=OBJECT)),
    dict(name="odezza_submit", description="Submit problem (states, known_rhs, trajectories), grammar and execution in one request. Or reuse problem_id. Returns a job handle before preparation/expansion; poll status/results. No automatic GP or LM.",
         inputSchema=SUBMISSION),
    dict(name="odezza_status", description="Read a submitted job's status and completed configuration count.", inputSchema=schema(["job_id"], job_id=STRING)),
    dict(name="odezza_results", description="Poll results: full report by default, or compact candidates once with leaderboard ID lists and replay addresses.",
         inputSchema=schema(["job_id"], job_id=STRING, format={"type": "string", "enum": ["full", "compact"]})),
    dict(name="odezza_cancel", description="Request cooperative cancellation between bounded native launches.", inputSchema=schema(["job_id"], job_id=STRING)),
    dict(name="odezza_replay", description="Resolve candidate_id, or reconstruct a retained score candidate by manifest/variant/configuration address using its exact coefficient snapshot. Optional FP64 CPU verification. Fitted results require candidate_id.",
         inputSchema=REPLAY),
    dict(name="odezza_fit", description="Start separate native LM on retained candidate IDs, preserving RNG, constants and state choices.",
         inputSchema=schema(["job_id", "request"], job_id=STRING, request=OBJECT, lanes={"type": "integer", "enum": [1, 2, 4, 8]})),
    dict(name="odezza_fit_results", description="Poll a separate LM operation.", inputSchema=schema(["fit_id"], fit_id=STRING)),
]


class Transport:
    def __init__(self, service):
        self.service, self.initialized, self.ready, self.fits = service, False, False, {}
        self.tools = [deepcopy(t) for t in TOOLS if
                      (not hasattr(service, "supported_tools") or t["name"] in service.supported_tools)
                      and (t["name"]!="odezza_release" or hasattr(service,"release"))]
        if hasattr(service,"supported_tools") and "odezza_fit" not in service.supported_tools:
            for tool in self.tools:
                if tool['name']=='odezza_prepare':
                    tool['description']='Store an optional raw problem in memory; C validates it after submission. Handles expire or can be released.'
                elif tool['name']=='odezza_submit':
                    tool['inputSchema']['properties'].pop('plan_only',None)
                elif tool['name']=='odezza_results':
                    tool['description']='Read compact candidates once with global/family/tag leaderboard references. Full is an alias. Export before releasing the job.'
                elif tool['name']=='odezza_replay':
                    tool['description']='Read a retained candidate by ID or family_index/ast_index/bank_index/permutation address with exact FP32 coefficients; optional CPU verification. Requires a live job handle.'

    def dispatch(self, message):
        identifier = message.get("id") if isinstance(message, dict) else None
        def error(code, text): return dict(jsonrpc="2.0", id=identifier, error=dict(code=code, message=text))
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return error(-32600, "Invalid JSON-RPC request")
        method, params = message["method"], message.get("params", {})
        if method == "notifications/initialized" and self.initialized:
            self.ready = True; return None
        if "id" not in message: return None
        if not isinstance(params, dict): return error(-32602, "params must be an object")
        if method == "initialize":
            if self.initialized: return error(-32600, "Already initialized")
            self.initialized = True
            result = dict(protocolVersion="2025-06-18", capabilities={"tools": {"listChanged": False}},
                          serverInfo={"name": "odezza-grammar", "version": "0.2.0"})
        elif method == "ping": result = {}
        elif not self.ready: return error(-32002, "Complete MCP initialization first")
        elif method == "tools/list": result = {"tools": self.tools}
        elif method == "tools/call":
            name, args = params.get("name"), params.get("arguments", {})
            tool = next((t for t in self.tools if t["name"] == name), None)
            if tool is None: return error(-32602, "Unknown tool")
            definition = tool["inputSchema"]
            if not isinstance(args, dict) or set(args)-set(definition["properties"]) or set(definition["required"])-set(args):
                return error(-32602, "Arguments do not match tool schema")
            if "oneOf" in definition and sum(set(choice["required"]) <= set(args) for choice in definition["oneOf"]) != 1:
                return error(-32602, "Provide exactly one of the alternative inputs")
            try:
                if name == "odezza_capabilities":
                    if hasattr(self.service, "capabilities"): value = self.service.capabilities()
                    else:
                        from .service import capabilities
                        value = capabilities()
                elif name == "odezza_fit":
                    import uuid
                    fit_id = uuid.uuid4().hex
                    self.fits[fit_id] = self.service.fit(**args)
                    value = dict(fit_id=fit_id, status="queued")
                elif name == "odezza_fit_results":
                    future = self.fits[args["fit_id"]]
                    value = future.result() if future.done() else dict(status="running", fit_id=args["fit_id"])
                elif name == "odezza_prepare": value = self.service.prepare(args["problem"])
                else: value = getattr(self.service, name.removeprefix("odezza_"))(**args)
                result = dict(content=[dict(type="text", text=json.dumps(value, allow_nan=False))], structuredContent=value, isError=False)
            except Exception as exc:
                code = getattr(exc,'code',None) or ('unknown_handle' if isinstance(exc,KeyError) else
                        'invalid_request' if isinstance(exc,(ValueError,TypeError)) else 'execution_error')
                value = dict(error=dict(code=code,message=str(exc)))
                result = dict(content=[dict(type="text", text=json.dumps(value))], structuredContent=value, isError=True)
        else: return error(-32601, "Method not found")
        return dict(jsonrpc="2.0", id=identifier, result=result)


def serve(service, source=None, output=None):
    source, output = source or sys.stdin, output or sys.stdout
    transport = Transport(service)
    while True:
        line = source.readline(MAX_MESSAGE_BYTES+1)
        if not line: break
        size = len(line) if isinstance(line,bytes) else len(line.encode('utf-8'))
        if size > MAX_MESSAGE_BYTES:
            # Drain this message in bounded pieces; keep the next request intact.
            while not line.endswith(b'\n' if isinstance(line,bytes) else '\n'):
                line = source.readline(MAX_MESSAGE_BYTES+1)
                if not line:break
            response=dict(jsonrpc="2.0",id=None,error=dict(code=-32600,
                          message="MCP message exceeds 16 MiB",data=dict(code="message_too_large",max_bytes=MAX_MESSAGE_BYTES)))
            output.write(json.dumps(response,separators=(",", ":"))+"\n");output.flush()
            continue
        try: response = transport.dispatch(json.loads(line))
        except (ValueError, TypeError): response = dict(jsonrpc="2.0", id=None, error=dict(code=-32700, message="Parse error"))
        if response is not None:
            output.write(json.dumps(response, allow_nan=False, separators=(",", ":"))+"\n"); output.flush()
