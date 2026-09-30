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

# SECANT Compile Matrix

Sources: [compile_matrix_2026-07-23_ptx.csv](compile_matrix_2026-07-23_ptx.csv), [compile_matrix_2026-07-23_cuda.csv](compile_matrix_2026-07-23_cuda.csv), [compile_matrix_2026-07-24_cubin.csv](compile_matrix_2026-07-24_cubin.csv)

Each cell reports aggregate AST/s. Module-shape columns are
`kernels/module x ASTs/kernel`. The faster backend for each module shape
is shown in bold.

For the PTX backend, NVRTC reduces the generated CUDA template to
optimized PTX once during handle creation, outside the measured hot path.
The timed compile therefore starts from this simpler template and covers
AST injection plus nvPTXCompiler. This preprocessing is also why
nvPTXCompiler O1 can occasionally compile SSE faster than O0: its cleanup
can reduce later lowering, register-allocation, and emission work by more
than the optimization passes cost.

![SECANT compile throughput](compile_matrix_2026-07-24.svg)

## 1 Worker

| Configuration | Backend | 8x24 | 16x24 | 32x24 | 64x8 | 64x16 | 64x64 |
|---|---|---:|---:|---:|---:|---:|---:|
| Materialize O0 ALU | PTX | **5,910** | **7,612** | **7,692** | **5,302** | **6,901** | **8,974** |
|  | CUDA | 2,293 | 2,425 | 2,447 | 1,920 | 2,278 | 1,329 |
| Materialize O0 MUFU | PTX | **3,373** | **3,718** | **3,780** | **3,020** | **3,528** | **4,057** |
|  | CUDA | 1,242 | 1,245 | 1,130 | 1,032 | 959 | 313 |
| Materialize O1 ALU | PTX | 5,008 | 5,170 | 5,313 | 3,442 | 4,689 | 6,213 |
|  | CUDA | 1,922 | 2,043 | 2,047 | 1,559 | 1,901 | 1,232 |
|  | CUBIN | **5,438,093** | **6,202,352** | **5,913,104** | **5,782,534** | **5,835,356** | **7,614,826** |
| Materialize O1 MUFU | PTX | 2,639 | 2,751 | 2,784 | 2,051 | 2,562 | 3,232 |
|  | CUDA | 1,077 | 1,084 | 994 | 862 | 848 | 305 |
|  | CUBIN | **2,731,618** | **2,783,093** | **2,633,636** | **2,595,711** | **2,856,238** | **2,849,500** |
| SSE O0 ALU | PTX | **988** | **986** | **993** | **981** | **1,005** | **898** |
|  | CUDA | 426 | 432 | 421 | 386 | 404 | 266 |
| SSE O0 MUFU | PTX | **831** | **831** | **844** | **839** | **862** | **761** |
|  | CUDA | 357 | 356 | 330 | 319 | 307 | 156 |
| SSE O1 ALU | PTX | 1,047 | 1,061 | 1,088 | 924 | 1,043 | 1,036 |
|  | CUDA | 406 | 408 | 402 | 358 | 381 | 261 |
|  | CUBIN | **6,588,880** | **7,095,672** | **7,728,456** | **6,993,723** | **7,565,795** | **7,714,917** |
| SSE O1 MUFU | PTX | 863 | 862 | 888 | 758 | 861 | 867 |
|  | CUDA | 337 | 338 | 316 | 290 | 287 | 152 |
|  | CUBIN | **2,842,213** | **2,933,806** | **2,929,833** | **2,973,529** | **2,972,234** | **2,918,405** |

## 12 Workers

| Configuration | Backend | 8x24 | 16x24 | 32x24 | 64x8 | 64x16 | 64x64 |
|---|---|---:|---:|---:|---:|---:|---:|
| Materialize O0 ALU | PTX | **57,766** | **70,997** | **76,015** | **50,233** | **68,414** | **88,738** |
|  | CUDA | 10,804 | 12,337 | 13,992 | 11,277 | 12,644 | 7,445 |
| Materialize O0 MUFU | PTX | **32,342** | **35,874** | **36,905** | **30,436** | **35,920** | **40,036** |
|  | CUDA | 6,511 | 6,354 | 5,841 | 5,660 | 4,699 | 1,429 |
| Materialize O1 ALU | PTX | 38,506 | 49,552 | 51,556 | 34,815 | 46,923 | 62,111 |
|  | CUDA | 10,153 | 12,587 | 12,526 | 10,529 | 12,018 | 7,423 |
|  | CUBIN | **12,553,634** | **16,877,141** | **19,674,168** | **17,608,621** | **23,624,885** | **34,797,679** |
| Materialize O1 MUFU | PTX | 25,870 | 28,079 | 28,519 | 21,106 | 26,346 | 32,312 |
|  | CUDA | 6,186 | 6,455 | 5,601 | 5,292 | 4,546 | 1,419 |
|  | CUBIN | **9,761,203** | **7,294,192** | **9,884,122** | **9,349,698** | **12,027,914** | **19,244,586** |
| SSE O0 ALU | PTX | **9,277** | **9,488** | **9,629** | **9,707** | **9,952** | **7,234** |
|  | CUDA | 3,225 | 3,305 | 3,262 | 2,991 | 3,120 | 2,121 |
| SSE O0 MUFU | PTX | **7,752** | **7,982** | **8,133** | **8,273** | **8,459** | **6,031** |
|  | CUDA | 2,682 | 2,597 | 2,399 | 2,351 | 2,103 | 1,005 |
| SSE O1 ALU | PTX | 10,260 | 10,439 | 10,754 | 9,437 | 10,466 | 9,980 |
|  | CUDA | 3,221 | 3,234 | 3,154 | 2,852 | 3,004 | 2,124 |
|  | CUBIN | **15,717,039** | **18,209,192** | **21,577,175** | **18,252,741** | **22,215,915** | **37,409,766** |
| SSE O1 MUFU | PTX | 8,229 | 8,540 | 8,817 | 7,597 | 8,648 | 8,314 |
|  | CUDA | 2,571 | 2,536 | 2,361 | 2,191 | 2,132 | 1,015 |
|  | CUBIN | **7,652,451** | **7,634,100** | **9,293,489** | **8,689,896** | **12,194,747** | **19,548,723** |

## 24 Workers

| Configuration | Backend | 8x24 | 16x24 | 32x24 | 64x8 | 64x16 | 64x64 |
|---|---|---:|---:|---:|---:|---:|---:|
| Materialize O0 ALU | PTX | **74,197** | **77,504** | **86,615** | **60,392** | **80,216** | **100,801** |
|  | CUDA | 10,905 | 13,498 | 14,738 | 12,345 | 13,824 | 8,426 |
| Materialize O0 MUFU | PTX | **39,761** | **42,432** | **43,320** | **35,628** | **41,625** | **45,636** |
|  | CUDA | 6,758 | 6,965 | 6,361 | 6,223 | 5,101 | 1,545 |
| Materialize O1 ALU | PTX | 53,601 | 61,347 | 64,460 | 41,853 | 58,570 | 74,048 |
|  | CUDA | 10,824 | 13,280 | 14,053 | 11,546 | 13,056 | 8,339 |
|  | CUBIN | **19,470,724** | **17,912,937** | **23,128,946** | **18,983,030** | **27,124,645** | **50,260,843** |
| Materialize O1 MUFU | PTX | 31,392 | 32,866 | 33,463 | 26,432 | 31,965 | 37,391 |
|  | CUDA | 6,865 | 7,055 | 6,294 | 5,962 | 5,056 | 1,540 |
|  | CUBIN | **8,231,151** | **16,874,995** | **12,885,028** | **10,520,048** | **14,201,348** | **19,437,419** |
| SSE O0 ALU | PTX | **10,074** | **10,468** | **10,390** | **10,999** | **11,081** | **7,129** |
|  | CUDA | 3,658 | 3,781 | 3,682 | 3,497 | 3,572 | 2,481 |
| SSE O0 MUFU | PTX | **8,517** | **8,636** | **8,690** | **9,580** | **9,439** | **5,961** |
|  | CUDA | 2,983 | 2,891 | 2,671 | 2,655 | 2,506 | 1,152 |
| SSE O1 ALU | PTX | 11,765 | 12,056 | 12,368 | 11,320 | 12,065 | 10,627 |
|  | CUDA | 3,581 | 3,648 | 3,679 | 3,349 | 3,545 | 2,495 |
|  | CUBIN | **14,204,511** | **19,709,786** | **13,481,346** | **20,847,241** | **28,866,195** | **53,518,537** |
| SSE O1 MUFU | PTX | 9,585 | 9,925 | 10,107 | 9,114 | 10,022 | 8,930 |
|  | CUDA | 2,919 | 2,816 | 2,645 | 2,498 | 2,399 | 1,161 |
|  | CUBIN | **7,350,554** | **10,407,315** | **12,742,724** | **11,229,445** | **13,626,357** | **21,077,881** |
