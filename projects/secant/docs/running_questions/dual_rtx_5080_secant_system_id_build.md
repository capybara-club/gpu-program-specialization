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

# Dual RTX 5080 Secant System ID node

**Decision date:** 2026-08-27  
**Purpose:** A reliable two-GPU rack node for independent Secant System ID
search jobs. This is a purchase plan, not a measured RTX 5080 result.

## Recommendation

Build one 5U AM5 node around two physically compact RTX 5080 cards. Run each
GPU as an independent queue target; do not synchronize a single search across
the cards. This shares the CPU, motherboard, memory, storage, network, and
power supply without constraining Secant's current workload, which uses about
1 GiB of VRAM per process and very little PCIe bandwidth on the hashed-settings
path.

Do not order both GPUs until the exact card model is known to fit and one RTX
5080 has passed the Secant acceptance benchmark. The expected throughput below
is projected from RTX 5090/4090 measurements and NVIDIA specifications; it has
not yet been measured on a 5080.

## Concrete parts list

| Component | Recommended part | Planning price | Reason |
| --- | --- | ---: | --- |
| GPUs | 2 × MSI GeForce RTX 5080 Shadow 3X OC 16 GB | 2 × $1,550 | A relatively compact 303 × 121 × 49 mm, 360 W card; avoid oversized premium coolers |
| CPU | AMD Ryzen 9 9950X | $549 | 16 cores/32 threads give comfortable specialization and job-control capacity for two GPUs without Threadripper cost |
| Motherboard | ASUS ProArt X870E-Creator WiFi | $495 | Two CPU-direct PCIe 5.0 slots that operate at x8/x8, plus integrated 10 GbE and 2.5 GbE |
| Memory | 32 GB, 2 × 16 GB, DDR5-6000 CL30 EXPO; for example Kingston Fury Beast `KF560C30BBEK2-32` | $400-$600 at the current abnormal market | Proven sufficient for the present two-GPU workload; do not pay the current four-figure price for 64 GB |
| Storage | Reputable 2 TB TLC NVMe drive in `M.2_1` | $250-$400 | Secant is not storage-bandwidth limited; do not pay a premium for PCIe 5.0 |
| Power supply | Seasonic Prime TX-1600 ATX 3.1 | $560 | 1,600 W, two native 12V-2x6 cables, and enough transient and sustained-load margin |
| Chassis | SilverStone RM51 5U rackmount | $450-$600 | Eight expansion slots and substantially safer GPU-height/power-cable clearance than 4U |
| Rack rails | SilverStone RMS05-22 or the RM51-compatible current rail kit | $100-$150 | Do not support a loaded 5U chassis only by its rack ears |
| CPU cooler | Noctua NH-D12L | $100 | 145 mm, AM5-compatible air cooler; avoids adding pump reliability and radiator obstruction |
| Airflow/accessories | Chassis fans as supplied, GPU support brackets, PWM control as needed | $100-$200 | Two open-air 360 W cards require deliberate front-to-back airflow and mechanical support |

**Planning total:** approximately **$6,100-$6,800 before tax and the rack
itself** at the prices observed on 2026-08-27. Memory and SSD pricing is
unusually inflated and volatile. The current process uses only about 300 MiB
of resident host memory, and complete hosts have remained below 4 GiB used, so
64 GB would add cost without measured throughput value. Upgrade later when
prices normalize or a larger dataset demonstrates a requirement.

The MSI card was listed by B&H at $1,549.99 but temporarily out of stock when
this note was written. Treat $1,600 per card as the purchase ceiling for this
plan. Recalculate the comparison if a different card or price is used.

## Lane and storage layout

Use a Ryzen 9000 desktop CPU, install the cards in the two CPU-connected x16
mechanical slots, and configure them as PCIe 5.0 x8/x8. PCIe 5.0 x8 per GPU is
ample for Secant's compact CUBIN, incumbent-setting, and result traffic.

The ProArt board's second GPU slot shares lanes with `M.2_2`. Leave `M.2_2`
empty; enabling it can reduce the second slot to x4. Put the system drive in
`M.2_1` and verify the negotiated link width under load after assembly. Do not
assume that a card being visible implies the intended lane configuration.

## Physical and thermal requirements

Two 49 mm cards nominally consume about five slots, but slot count alone is not
enough. Before purchase, verify all of the following against the exact card and
chassis drawings:

1. both motherboard slot centerlines and card backplate thickness;
2. intake clearance for the upper card;
3. at least the GPU vendor's required clearance for each 12V-2x6 cable, with no
   sharp bend at the connector;
4. card length, height, and support-bracket clearance;
5. unobstructed front intake and rear exhaust; and
6. rack depth, rail compatibility, and a dedicated power circuit.

The two GPUs have a combined 720 W board-power rating. With a 170 W CPU and
the rest of the host, plan for roughly 850-1,000 W at the wall under a sustained
Secant load. A dedicated 120 V / 15 A circuit is advisable for this node; do
not place two such nodes on the same ordinary circuit. Measure actual wall
power rather than extrapolating from gaming benchmarks.

If sustained dual-GPU testing shows the upper card throttling, first improve
fan curves and chassis airflow, then test a modest GPU power limit. If each
card cannot retain at least 95% of its solo Secant throughput, use two
single-GPU hosts instead. The modest host-cost saving is not worth a permanent
thermal penalty.

## Expected Secant behavior

NVIDIA specifies 10,752 CUDA cores, 16 GB GDDR7, and 360 W for one RTX 5080.
Two cards have almost the RTX 5090's aggregate CUDA-core count, but they are two
independent devices with separate module-loading and launch pipelines.

The provisional expectation is **65-80 million direct settings/s per card**,
or **130-160 million settings/s across two independent jobs**. Rohini's RTX
5090 measured 145.5 million direct settings/s. These are planning numbers,
not a claim that two 5080s equal a 5090 for every Secant kernel.

Sixteen gigabytes of VRAM is ample. The current efficient search uses about 1
GiB total per GPU process, with only roughly 12 MiB attributable to explicit
trajectory, setting, solver, and result buffers. The workload is selected by
compute throughput, registers, shared memory, module behavior, cooling, and
price—not VRAM capacity.

## Acceptance test before replication

Test one GPU alone and then both GPUs concurrently with the same clocks and
software revision. Record:

1. direct settings/s;
2. packed GP configurations/s;
3. trajectory-LM fits/s for the production settings/start count;
4. C99 specialization and eager module-loading rates;
5. negotiated PCIe generation and width for both cards;
6. GPU temperature, hotspot temperature, clock, power, and throttling reason;
7. wall power; and
8. per-GPU throughput while the other device is simultaneously saturated.

Accept the node for replication only if each card reaches at least **72 million
direct settings/s**, both retain at least **95% of solo throughput** under
concurrent load, and a multi-hour test has no thermal throttling, CUDA errors,
or materialized-replay failures.

## Software and dispatch policy

- Give each GPU its own persistent Secant worker/loader queue and device ID.
- Schedule complete seeds, datasets, or recovery problems to one GPU.
- Keep the fixed CUDA template and CUBIN cache local to the node.
- Let a lightweight central queue assign jobs; no NVLink or peer-to-peer GPU
  communication is required.
- Record GPU model, driver, architecture, commit, kernel mode, settings, starts,
  power limit, and acceptance benchmark in every result.

## Sources

- [NVIDIA RTX 5080 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/)
- [MSI RTX 5080 Shadow 3X OC dimensions and power](https://us.msi.com/Graphics-Card/GeForce-RTX-5080-16G-SHADOW-3X-OC/Specification)
- [Current B&H MSI RTX 5080 Shadow listing](https://www.bhphotovideo.com/c/product/1879047-REG/msi_g5080_16s3c_geforce_rtx_5080_16g.html/overview)
- [AMD Ryzen 9 9950X specifications](https://www.amd.com/en/products/processors/desktops/ryzen-9000-series/amd-ryzen-9-9950x.html)
- [Current B&H Ryzen 9 9950X listing](https://www.bhphotovideo.com/c/product/1834127-REG/amd_100_100001277wof_ryzen_9_9950x.html)
- [ASUS ProArt X870E-Creator manual and x8/x8 sharing rules](https://dlcdnets.asus.com/pub/ASUS/mb/SocketAM5/ProArt_X870E-CREATOR_WIFI/E23930_ProArt_X870E-CREATOR_WIFI_EM_WEB.pdf?model=ProArt+X870E-CREATOR+WIFI)
- [Current B&H ProArt X870E-Creator listing](https://www.bhphotovideo.com/c/product/1855749-REG/asus_proart_x870e_creator_wifi_motherboard.html)
- [Seasonic Prime TX-1600 ATX 3.1 specifications](https://seasonic.com/atx3-prime-tx/)
- [Current B&H Seasonic Prime TX-1600 listing](https://www.bhphotovideo.com/c/product/803373802-USE/seasonic_electronics_prime_tx_1600_atx3_1_prime_tx_atx_3_1.html)
- [SilverStone RM51 product sheet](https://gzhls.at/blob/ldb/f/b/3/d/df764bb8e8c62cca944822f1e35ce4e7cf6b.pdf)
- [Noctua NH-D12L specifications](https://noctua.at/en/products/cpu-cooler-retail/nh-d12l)
- [Kingston DDR5-6000 CL30 EXPO family](https://www.kingston.com/en/memory/gaming/kingston-fury-beast-ddr5-memory)
- [August 2026 DDR5 market-price survey](https://www.tomshardware.com/pc-components/ram/memory-prices-climb-500-percent-in-12-months-up-to-10x-the-lowest-ever-tracked-prices-128gb-of-ddr5-now-usd3-399)
