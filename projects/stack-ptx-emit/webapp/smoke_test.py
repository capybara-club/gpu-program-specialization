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

from app import create_app


def main() -> int:
    app = create_app()
    app.testing = True
    client = app.test_client()

    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}

    response = client.get("/api/v1/functions")
    assert response.status_code == 200
    function_names = [entry["name"] for entry in response.get_json()["functions"]]
    assert "fma" in function_names
    assert "div" in function_names

    response = client.post(
        "/api/v1/linear-regression/generate",
        json={
            "seed": 5,
            "input_dim": 4,
            "num_features": 3,
            "steps": 8,
            "eps": 1e-5,
            "allowed_functions": ["abs", "add", "mul", "fma"],
            "backends": ["c", "markdown"],
            "function_name": "web_infer",
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["request"]["function_name"] == "web_infer"
    assert abs(payload["request"]["eps"] - 1e-5) < 1e-12
    assert payload["request"]["allowed_functions"] == ["abs", "add", "mul", "fma"]
    assert set(payload["outputs"]) == {"c", "markdown"}

    response = client.post(
        "/",
        data={
            "seed": "5",
            "input_dim": "4",
            "num_features": "3",
            "steps": "8",
            "eps": "1e-5",
            "allowed_functions": ["sin", "cos", "div", "fma"],
            "function_name": "web_infer",
            "markdown_factor_expression_min_bytes": "48",
            "backends": ["c", "markdown"],
        },
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Backend Outputs" in html
    assert "web_infer" in html
    assert "Feature Programs" in html
    assert "Beta Unstandardized" in html
    assert "Protected Epsilon" in html
    assert "Allowed Functions" in html
    assert 'class="markdown-document"' in html
    assert "Linear Regression Model Specification" in html
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
