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

# Internal implementation contract

Python 3.10+ standard library only. Package `odegrammar`; no ODE/GPU execution. Stream JSONL metadata and programs. User input is JSON with safe infix strings.

Canonical input: version=1, states=[x0,...], integration={method:rk4,dt:positive,stiff:false}, rules={"R(z)":[strings or {expr,tags,locals}]}, shapes={name:expression|tagged-alternative|alternative-list|{"choices":[...]}}, rhs={state:expression}. Alternatively families=[{id,tags,rhs,rules,shapes,leaves,constants,rng,parameters}] inheriting top-level definitions. All RHS components required. Top-level constant_banks={name:[finite numbers]}, leaves={axis:{states:[state names],arity:2|4,coverage:all|sample|explicit,samples:N,seed:int,groups:[[states...]]}}, constants={slot:{bank:name}|{values:[numbers]}|{value:number}}, rng_banks={name:{base:uniform01|normal01,count:int,seed:int,scope:run|skeleton}}, rng={slot:{bank:name,axis:optional shared axis,stream:optional,transform:{kind:identity|affine|uniform|normal|log_uniform,...}}}, parameters={name:{initial:number}}. Named axis references share selection/draw. Missing rng axis defaults to slot-specific independent axis. Same shared axis means zipped bank indices, equal counts required. Independent axes form Cartesian products. Same stream and bank deliberately share raw draws.

Rules support hole(R,args...), bare zero-argument rule references, named shape.f, tagged alternatives. `locals` contains leaves/constants/rng/parameters scoped to an alternative, with hygienic reference renaming. Explicit global names stay shared. Tags are AST wrappers and preserve spans in final postorder.

expr.py API: frozen Node(op:str,value:object=None,children:tuple[Node,...]=()); parse_expr(text)->Node; substitute(node,mapping[str,Node])->Node; rename_refs(node,mapping[tuple(namespace,name),str])->Node; node_count/node_depth ignore tag wrappers; walk(node). Node ops: literal(value number), symbol(value bare name), leaf/const/rng/param/shape(value slot name), hole(value rule name,children actual arguments), add/sub/mul/div(children2), neg(children1), powi(value integer,children1), sin/cos/tanh/exp/log/sqrt/abs(children1), tag(value tuple[str,...],children1). Grammar signatures parsed separately; RHS bare xN remains symbol until lowering. `t` is accepted by the compiler and lowers to TIME; the execution backend must separately support it.

expansion.py API: ExpandedSystem(rhs:dict[str,Node], locals:dict[str,dict]); GrammarExpander(rules,shapes,max_nodes=63,max_depth=12,max_expansion_depth=20,max_steps=1000000,deadline=None). expand(rhs:dict[str,str|Node],strategy='enumerate'|'sample',seed=0,max_derivations=1000000)->iterator[ExpandedSystem]. .stats dict; .stop_reason str|None. Independent hole occurrences fresh; named shapes selected once per system. locals keys leaves/constants/rng/parameters; generated names prefixed __local_ for compiler normalization. Tags on rule alternatives wrap their expanded subtree. No AST dedup in expander: compiler preserves provenance and deduplicates lowered programs. Validate references/cycles or bound recursion. Use lazy products, do not list expansions. One scalar rule symbol can stand for bare zero-argument rule. Parser errors should be descriptive ValueError subclasses.

pools.py API: PoolPlanner(states,leaves,constants,constant_banks,rng,rng_banks,parameters,seed=0). iter_variants(active:dict[str,set[str]],skeleton_id:str,group_sampling:dict|None=None)->iterator[dict]. active keys leaf,const,rng,param. Each output has toggles (slot->list state indices), pool_axes (ordered list {id,kind,count,...}), constants metadata, rng_bindings metadata, prelude list of per-slot setup instruction descriptors, rng_bank_requests, parameter_initials, configuration_count integer. Only active bindings consume axes; all declared definitions should be structurally validated. Toggle coverage all uses combinations(states,arity), combinations of groups across independent named slots; sample uses N unique groups without replacement and deterministic seed (no replacement if N exceeds finite space: cap to exhaustive); explicit groups exact arity/distinct states. Reject fewer distinct states than arity. State order normalized by declared state index. No 3-way runtime op or duplicate padding. Fixed states use xN directly. Repeated axis uses same selected group/choice. Overlapping groups legitimately repeat scalar candidates; count configuration visits honestly.

RNG prelude: standard banks generated separately; per configuration each binding reads selected bank index once, then applies transform before RHS loop. Bank generation is not implemented as CUDA: emit reproducible bank descriptors and optional Python reference materialization with documented non-Philox test RNG. Addressing contract must be stable, independent of scheduling and variant order. Run scope same raw bank across skeletons; skeleton scope keyed by skeleton_id. Stream default binding name; same explicit stream shares raw values. base uniform01 maps uniform or log_uniform; base normal01 maps normal. affine accepts either. Params of transforms can be finite numbers or const.slot references (validate slot and activate dependencies). Prelude placeholder RHS RNG_VALUE keeps transform out of integration loop.

Integration: current executable registry only rk4. Unknown/future method allowed only with explicit annotation-only option, yielding integration.backend_supported=false and integration.annotation_only=true (there is no executable field); never silently use rk4 for requested stiff integrator. stiff=true with rk4 rejected unless explicit future annotation mode, marked incompatible. This compiler emits method metadata and does not implement integration.

Separate LM request compiler: kind=lm, states array <=8, candidate IDs, parameters distinct list <=8 naming fitted RHS scalars, numerical LM controls; return metadata only. Grammar parameter declarations accept finite numerical initial values; the separate LM request does not carry those values or initial states. Initial-condition fitting is unsupported. Main compile request cannot request LM execution. Counting non-fitted const/rng slots independent of fitted dimension.

The CLI is in odegrammar/__main__.py; orchestration/lowering is in odegrammar/compiler.py. Output header, skeleton records (whole RHS dict postorder; program ID includes leaf kinds/arities, op values and parameter sharing but excludes tags/runtime pool values), variant records with pool/prelude metadata, provenance events for duplicates, final summary. Split structural program vs toggle group variants. Limits separately max_skeletons/max_variants/max_configurations/max_derivations/max_expansion_steps/max_seconds. Retain metadata global/per_family/by_tag carried through for downstream top-k. Compiler provides optional reference evaluation helpers for checking outputs, no numerical solver claims.

Bounds: expansion.max_nodes limits the sum of nodes across the entire RHS vector; max_depth limits each RHS. Shared shapes are inlined and count at every occurrence. Skeleton active_slots lists direct RHS references; allocate transform-only constant dependencies from the complete variant pools. See ../INTEGRATION_HANDOFF.md for execution identity, RNG profile, provenance and backend-capability requirements.
