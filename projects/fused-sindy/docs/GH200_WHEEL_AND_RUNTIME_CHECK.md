<!--
SPDX-FileCopyrightText: 2026 Charles Durham
SPDX-License-Identifier: MIT

MIT License

Copyright (c) 2026 Charles Durham

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# GH200 Wheel And Runtime Check

This was run before terminating the GH200 instance.

## Clean Wheel Install

A fresh venv was created with `--system-site-packages` so it could reuse the system PyTorch install. Older runs required an external `cubin-function-patch` checkout.

The header is now vendored in `thirdparty/`, so the equivalent current command is:

```bash
python -m pip wheel --no-build-isolation -w dist .
```

Historical result before the project distribution was renamed:

- Built `implicit_sindy-0.1.0-cp312-cp312-linux_aarch64.whl`
- Installed into the fresh venv
- Imported from site-packages
- PyTorch CUDA available on `NVIDIA GH200 480GB`
- `test_pytorch_gram_nanobind.py`: passed 2/2
- `test_pytorch_pysindy_example.py`: passed 1/1

## Runtime Benchmark

Historical standalone Gram-runtime benchmark summary from the GH200 instance.
The old bench included an explicit `folds=1` option; the current API has
removed internal fold aggregation and should be benchmarked through the
three-launch `gram -> solve -> mse` path.

Result summary:

- Compile backend: `ast-patch`
- Compile wall time: `2474.334 ms`
- Settings: `2048`
- Rows per setting: `131072`
- Timed launches: `5`
- Average grid launch time: `428.178 ms`
- Settings/sec: `4783.055`
- Rows/sec: `626,924,588`
- Feature rows/sec: `20,061,586,829`

Raw historical output: `gh200_runtime_bench_2048_settings_131072_rows.txt`
