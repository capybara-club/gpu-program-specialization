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

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import markdown as markdown_lib
from flask import Flask, jsonify, render_template, request


SUPPORTED_BACKENDS = ["c", "cuda", "rust", "numpy", "latex", "markdown"]
SUPPORTED_FUNCTIONS = [
    {"name": "abs", "arity": 1},
    {"name": "neg", "arity": 1},
    {"name": "rcp", "arity": 1},
    {"name": "sqrt", "arity": 1},
    {"name": "rsqrt", "arity": 1},
    {"name": "sin", "arity": 1},
    {"name": "cos", "arity": 1},
    {"name": "log2", "arity": 1},
    {"name": "exp2", "arity": 1},
    {"name": "tanh", "arity": 1},
    {"name": "add", "arity": 2},
    {"name": "sub", "arity": 2},
    {"name": "mul", "arity": 2},
    {"name": "div", "arity": 2},
    {"name": "min", "arity": 2},
    {"name": "max", "arity": 2},
    {"name": "copysign", "arity": 2},
    {"name": "fma", "arity": 3},
]
SUPPORTED_FUNCTION_NAMES = [entry["name"] for entry in SUPPORTED_FUNCTIONS]

DEFAULT_FORM_STATE = {
    "seed": 7,
    "input_dim": 8,
    "num_features": 6,
    "steps": 12,
    "eps": 1.0e-6,
    "function_name": "linear_regression_infer",
    "input_names": "",
    "show_spec": False,
    "markdown_factor_expression_min_bytes": 96,
    "allowed_functions": SUPPORTED_FUNCTION_NAMES,
    "backends": SUPPORTED_BACKENDS,
}


def _default_cli_path() -> Path:
    env_path = os.environ.get("STACK_PTX_EMIT_LINEAR_REGRESSION_CLI")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parent.parent / "build" / "tools" / "stack_ptx_emit_linear_regression_cli"


def _coerce_int(raw: Any, name: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _coerce_float(raw: Any, name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _coerce_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _optional_text(raw: Any, default: str = "") -> str:
    if raw is None:
        return default
    return str(raw).strip()


def _normalize_backends(raw: Any) -> list[str]:
    if raw is None:
        return list(SUPPORTED_BACKENDS)
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        try:
            items = [str(item).strip() for item in raw if str(item).strip()]
        except TypeError as exc:
            raise ValueError("backends must be a list or comma-separated string") from exc
    if not items:
        raise ValueError("at least one backend must be selected")
    invalid = [item for item in items if item not in SUPPORTED_BACKENDS]
    if invalid:
        raise ValueError(f"unsupported backends: {', '.join(invalid)}")
    deduped: list[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _normalize_allowed_functions(raw: Any) -> list[str]:
    if raw is None:
        return list(SUPPORTED_FUNCTION_NAMES)
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        try:
            items = [str(item).strip() for item in raw if str(item).strip()]
        except TypeError as exc:
            raise ValueError("allowed_functions must be a list or comma-separated string") from exc
    if not items:
        raise ValueError("at least one function must be selected")
    invalid = [item for item in items if item not in SUPPORTED_FUNCTION_NAMES]
    if invalid:
        raise ValueError(f"unsupported functions: {', '.join(invalid)}")
    if not any(item in {"add", "sub", "mul", "div", "min", "max", "copysign", "fma"} for item in items):
        raise ValueError("allowed_functions must include at least one binary or ternary operator")
    deduped: list[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _normalize_request(raw: dict[str, Any]) -> dict[str, Any]:
    payload = dict(DEFAULT_FORM_STATE)
    payload["seed"] = _coerce_int(raw.get("seed", payload["seed"]), "seed")
    payload["input_dim"] = _coerce_int(raw.get("input_dim", payload["input_dim"]), "input_dim")
    payload["num_features"] = _coerce_int(raw.get("num_features", payload["num_features"]), "num_features")
    payload["steps"] = _coerce_int(raw.get("steps", payload["steps"]), "steps")
    payload["eps"] = _coerce_float(raw.get("eps", payload["eps"]), "eps")
    payload["markdown_factor_expression_min_bytes"] = _coerce_int(
        raw.get(
            "markdown_factor_expression_min_bytes",
            payload["markdown_factor_expression_min_bytes"],
        ),
        "markdown_factor_expression_min_bytes",
    )
    payload["function_name"] = _optional_text(raw.get("function_name"), payload["function_name"]) or payload["function_name"]
    payload["input_names"] = _optional_text(raw.get("input_names"), "")
    payload["show_spec"] = _coerce_bool(raw.get("show_spec", payload["show_spec"]))
    payload["allowed_functions"] = _normalize_allowed_functions(
        raw.get("allowed_functions", payload["allowed_functions"])
    )
    payload["backends"] = _normalize_backends(raw.get("backends", payload["backends"]))
    return payload


def _build_cli_command(payload: dict[str, Any], cli_path: Path) -> list[str]:
    command = [
        str(cli_path),
        "--seed",
        str(payload["seed"]),
        "--input-dim",
        str(payload["input_dim"]),
        "--num-features",
        str(payload["num_features"]),
        "--steps",
        str(payload["steps"]),
        "--eps",
        str(payload["eps"]),
        "--allowed-functions",
        ",".join(payload["allowed_functions"]),
        "--function-name",
        payload["function_name"],
        "--backends",
        ",".join(payload["backends"]),
        "--markdown-factor-bytes",
        str(payload["markdown_factor_expression_min_bytes"]),
    ]
    if payload["input_names"]:
        command.extend(["--input-names", payload["input_names"]])
    if payload["show_spec"]:
        command.append("--show-spec")
    return command


def _run_generator(payload: dict[str, Any], cli_path: Path) -> dict[str, Any]:
    if not cli_path.exists():
        raise RuntimeError(f"generator binary not found: {cli_path}")

    command = _build_cli_command(payload, cli_path)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "generator failed"
        raise RuntimeError(stderr)

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("generator returned invalid JSON") from exc


def _build_backend_views(outputs: dict[str, str]) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for backend, text in outputs.items():
        rendered_html = None
        if backend == "markdown":
            rendered_html = markdown_lib.markdown(
                text,
                extensions=["tables", "md_in_html", "sane_lists"],
            )
        views.append(
            {
                "backend": backend,
                "text": text,
                "rendered_html": rendered_html,
                "is_markdown_document": backend == "markdown",
            }
        )
    return views


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["STACK_PTX_EMIT_LINEAR_REGRESSION_CLI"] = str(_default_cli_path())

    @app.get("/healthz")
    def healthz() -> Any:
        return jsonify({"ok": True})

    @app.get("/api/v1/backends")
    def list_backends() -> Any:
        return jsonify({"backends": SUPPORTED_BACKENDS})

    @app.get("/api/v1/functions")
    def list_functions() -> Any:
        return jsonify({"functions": SUPPORTED_FUNCTIONS})

    @app.post("/api/v1/linear-regression/generate")
    def generate_api() -> Any:
        raw = request.get_json(silent=True)
        if raw is None:
            return jsonify({"error": "expected JSON request body"}), 400

        try:
            payload = _normalize_request(raw)
            result = _run_generator(
                payload,
                Path(app.config["STACK_PTX_EMIT_LINEAR_REGRESSION_CLI"]),
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500

        return jsonify(result)

    @app.route("/", methods=["GET", "POST"])
    def index() -> Any:
        form_state = dict(DEFAULT_FORM_STATE)
        result = None
        error = None

        if request.method == "POST":
            raw = {
                "seed": request.form.get("seed"),
                "input_dim": request.form.get("input_dim"),
                "num_features": request.form.get("num_features"),
                "steps": request.form.get("steps"),
                "eps": request.form.get("eps"),
                "function_name": request.form.get("function_name"),
                "input_names": request.form.get("input_names"),
                "show_spec": request.form.get("show_spec"),
                "allowed_functions": request.form.getlist("allowed_functions"),
                "markdown_factor_expression_min_bytes": request.form.get(
                    "markdown_factor_expression_min_bytes"
                ),
                "backends": request.form.getlist("backends"),
            }
            try:
                form_state = _normalize_request(raw)
                result = _run_generator(
                    form_state,
                    Path(app.config["STACK_PTX_EMIT_LINEAR_REGRESSION_CLI"]),
                )
            except (ValueError, RuntimeError) as exc:
                error = str(exc)
                form_state = {
                    **form_state,
                    **{
                        "seed": raw.get("seed") or DEFAULT_FORM_STATE["seed"],
                        "input_dim": raw.get("input_dim") or DEFAULT_FORM_STATE["input_dim"],
                        "num_features": raw.get("num_features") or DEFAULT_FORM_STATE["num_features"],
                        "steps": raw.get("steps") or DEFAULT_FORM_STATE["steps"],
                        "eps": raw.get("eps") or DEFAULT_FORM_STATE["eps"],
                        "function_name": raw.get("function_name") or DEFAULT_FORM_STATE["function_name"],
                        "input_names": raw.get("input_names") or "",
                        "show_spec": _coerce_bool(raw.get("show_spec")),
                        "allowed_functions": raw.get("allowed_functions") or list(DEFAULT_FORM_STATE["allowed_functions"]),
                        "markdown_factor_expression_min_bytes": raw.get("markdown_factor_expression_min_bytes")
                        or DEFAULT_FORM_STATE["markdown_factor_expression_min_bytes"],
                        "backends": raw.get("backends") or list(DEFAULT_FORM_STATE["backends"]),
                    },
                }

        return render_template(
            "index.html",
            defaults=DEFAULT_FORM_STATE,
            form_state=form_state,
            supported_backends=SUPPORTED_BACKENDS,
            supported_functions=SUPPORTED_FUNCTIONS,
            result=result,
            backend_views=_build_backend_views(result["outputs"]) if result is not None else None,
            error=error,
            cli_path=app.config["STACK_PTX_EMIT_LINEAR_REGRESSION_CLI"],
        )

    return app


app = create_app()


if __name__ == "__main__":
    host = os.environ.get("STACK_PTX_EMIT_WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("STACK_PTX_EMIT_WEB_PORT", "8000"))
    debug = _coerce_bool(os.environ.get("STACK_PTX_EMIT_WEB_DEBUG", "true"))
    app.run(host=host, port=port, debug=debug)
