#!/usr/bin/env python3
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
"""Bounded first search and grouped feedback, keeping the supplied game grammar.

Uses the existing trial service. Does not run a second search automatically.
"""
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import time

import ode_game as game


def size_counts(config):
    """Exact ordered-AST counts by node count, before numerical invalidity."""
    game.validate(config)
    g = config['grammar']; levels = [{1: config['states']+1}]
    for _ in range(g['max_depth']):
        previous = levels[-1]; current = {1: config['states']+1}
        for n, count in previous.items():
            current[n+1] = current.get(n+1, 0)+len(g['unary'])*count
            for m, other in previous.items():
                current[n+m+1] = current.get(n+m+1, 0)+len(g['binary'])*count*other
        levels.append({n:c for n,c in current.items() if c})
    return levels


def allocate(populations, budget):
    """Equal allocation with unused small-stratum quotas redistributed."""
    if budget < len(populations): raise ValueError('AST budget must cover every nonempty stratum')
    quotas = [0]*len(populations); remaining = min(budget, sum(populations))
    while remaining:
        active = [i for i,p in enumerate(populations) if quotas[i] < p]
        share = max(1, remaining//len(active))
        for i in active:
            take = min(share, populations[i]-quotas[i], remaining)
            quotas[i] += take; remaining -= take
            if not remaining: break
    return quotas


def prepare(config, ast_budget=65536, banks=768, search_seed=0, seconds=90, target_mse=None):
    game.integer(ast_budget, 1, 10_000_000, 'AST budget')
    if config['grammar']['max_depth'] > 3:
        raise ValueError('The bounded probe currently supports grammar depth 0 through 3')
    fragment, population = game.compile_search(config, banks, search_seed)
    original = fragment['grammar_search']; levels = size_counts(config)
    depth = config['grammar']['max_depth']; families = []
    for family in original['families']:
        op = family['name'][5:]
        if op in config['grammar']['unary'] and depth:
            sizes = {n+1:c for n,c in levels[-2].items()}
        elif op in config['grammar']['binary'] and depth:
            sizes = {}
            for n,c in levels[-2].items():
                for m,d in levels[-2].items(): sizes[n+m+1] = sizes.get(n+m+1,0)+c*d
        else: sizes = {1:1}
        # Small sizes share a bucket; larger sizes stay separate. This keeps
        # three-state depth3's strata within the service's 64-family limit and
        # puts 13/14/15-node trees into the first request, not behind a prefix.
        buckets = {}
        for n,count in sizes.items():
            key = (min(sizes), min(4,max(sizes))) if n <= 4 else (n,n)
            buckets[key] = buckets.get(key,0)+count
        for (lo,hi), count in sorted(buckets.items()):
            f = copy.deepcopy(family)
            f.update(name=family['name']+f'-nodes-{lo}-{hi}', minimum_nodes=lo, maximum_nodes=hi)
            f['tags'] = dict(f['tags'], parent_family=family['name'], population=str(count),
                            node_range=[lo,hi], coverage='bounded_enumeration_prefix_within_stratum')
            families.append(f)
    if len(families) > 64: raise ValueError('Probe partition exceeds the existing 64-family limit')
    totals = [int(f['tags']['population']) for f in families]
    assert sum(totals) == int(population['skeletons_per_rhs'])
    for f,quota in zip(families, allocate(totals,ast_budget)):
        f.update(accepted=quota,work_limit=max(10000,quota*1000))
        if quota == int(f['tags']['population']): f['tags']['coverage'] = 'exhaustive_requested_within_stratum'
    original['families'] = families
    request = game.campaign(fragment, seconds, target_mse=game.target_for(config,target_mse))
    request['policy_options'].update(family_survivors=1, global_survivors=64,
                                     maximum_survivors=128, round_decimals=[])
    # Let the service return to the caller promptly after it has exhausted this
    # bounded request. Future requests are proposed from actual feedback.
    available = seconds-10
    # Keep bank-size changes from silently selecting the much larger 512-AST
    # kernel. The first-report calibration uses the established 256 capacity.
    request['workflow'] = dict(screen={'wall_seconds':available*.5,'system_capacity':256},
        refine={'wall_seconds':available*.3},full={'wall_seconds':available*.2,'candidates':16},
        revision_wait_seconds=0)
    validate_budget(request)
    coverage = dict(grammar_sha256=game.digest(config), compiled_grammar_sha256=game.digest(original['grammar']),
        grammar_unchanged=True, full_language_asts=str(sum(totals)),
        requested_asts=sum(f['accepted'] for f in families), bank_rows=banks,
        requested_configurations=sum(f['accepted'] for f in families)*banks,
        partitions=[dict(name=f['name'],start=f['start'],**f['tags'],requested=f['accepted']) for f in families],
        interpretation='Disjoint root/node-count strata partition the original language. Within each stratum the native enumerator takes a prefix, not a uniform random sample. No operators, depth, states or constants were changed.')
    return request, coverage


def validate_budget(request):
    budget=request['budget']; workflow=request['workflow']
    reserve=game.finite(budget['verification_reserve_seconds'],'verification reserve',True)
    wall=game.finite(budget['wall_seconds'],'wall seconds',True)
    used=sum(game.finite(workflow[p]['wall_seconds'],p+' seconds',True) for p in ('screen','refine','full'))
    wait=game.finite(workflow.get('revision_wait_seconds',0),'revision wait')
    if wait < 0 or used+wait+reserve > wall+1e-8: raise ValueError('Phase budgets exceed total wall budget')


def shape(expression, states):
    """Ordered structural signature; no algebraic-equivalence claim."""
    tree=ast.parse(expression,mode='eval').body; slots={}
    def visit(node):
        if isinstance(node,ast.Name):
            if node.id in states: return node.id,0,1
            slots.setdefault(node.id,len(slots));return 'p'+str(slots[node.id]),0,1
        if isinstance(node,ast.Constant): return repr(node.value),0,1
        if isinstance(node,ast.UnaryOp):
            s,d,n=visit(node.operand);return type(node.op).__name__+'('+s+')',d+1,n+1
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and len(node.args)==1:
            s,d,n=visit(node.args[0]);return node.func.id+'('+s+')',d+1,n+1
        if isinstance(node,ast.BinOp):
            a,ad,an=visit(node.left);b,bd,bn=visit(node.right)
            return type(node.op).__name__+'('+a+','+b+')',1+max(ad,bd),1+an+bn
        raise ValueError('Unsupported expression in report')
    signature,depth,nodes=visit(tree)
    children = [visit(tree.left)[1],visit(tree.right)[1]] if isinstance(tree,ast.BinOp) else []
    root=type(tree.op).__name__ if isinstance(tree,ast.BinOp) else tree.func.id if isinstance(tree,ast.Call) else 'leaf'
    return dict(signature=signature,depth=depth,nodes=nodes,root=root,child_depths=children,
                active_states=sorted({n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id in states}))


def summarize(state, states):
    grouped={}
    for scope,rows in state.get('frontiers',{}).items():
        groups={}
        for row in rows:
            if row.get('mse') is None or not math.isfinite(row['mse']): continue
            feature=shape(row['expression'],states)
            # Never merge scores across objectives, even within a named phase.
            key=(row.get('objective_id'),feature['signature'])
            item=groups.setdefault(key,dict(**feature,objective_id=row.get('objective_id'),records=0,
                families=set(),best_record=row))
            item['records']+=1;item['families'].update(row.get('families',[]))
            if row['mse'] < item['best_record']['mse']:item['best_record']=row
        grouped[scope]=[dict(v,families=sorted(v['families'])) for v in sorted(groups.values(),
            key=lambda v:(str(v['objective_id']),v['best_record']['mse']))]
    return dict(status=state['status'],campaign_id=state.get('campaign_id'),groups_by_scope=grouped,
        grouping='Ordered AST structure with parameter renaming; repeated fits collapse, commuted or algebraically equivalent structures may remain separate. Complete best records retain replay provenance.',
        coverage=state.get('grammar_coverage',{}),families=state.get('families',{}),
        family_coverage=state.get('family_coverage',{}),
        timing=state.get('timing',{}),phase_seconds=state.get('phase_seconds',{}),
        coefficient_trials=state.get('coefficient_trials'),verification=state.get('verification'))


def recommend(state, request, coverage, feedback_seconds=5):
    screen=state.get('phase_seconds',{}).get('screen',0)
    achieved=sum(f.get('evaluated_model_occurrences',0) for f in state.get('grammar_coverage',{}).values())
    actual=state.get('grammar_coverage',{})
    complete=(state.get('accounting_complete',False) and achieved==coverage['requested_asts']
        and all(actual.get(p['name'],{}).get('evaluated_model_occurrences')==p['requested']
                for p in coverage['partitions']))
    if state['status']=='verified': return dict(action='stop_verified')
    if not complete or screen<=0:
        return dict(action='inspect_incomplete_work',reason='Do not extrapolate throughput from incomplete accounting or missed quotas.')
    estimate=round(achieved*feedback_seconds/screen)
    count=min(int(coverage['full_language_asts']),10_000_000,max(len(coverage['partitions']),
              max(achieved//2,min(achieved*4,estimate))))
    if count <= achieved:
        return dict(action='review_groups_for_refinement',measured_asts=achieved,
            measured_screen_seconds=screen,bank_rows=coverage['bank_rows'],
            reason='Repeating the same seed and a smaller or equal prefix adds no screen coverage. Review retained fits or deliberately choose new coefficient rows.')
    return dict(action='review_groups_then_choose_next_request',suggested_ast_budget=count,
        bank_rows=coverage['bank_rows'],target_screen_seconds=feedback_seconds,measured_screen_seconds=screen,
        measured_asts=achieved,estimated_screen_seconds=screen*count/achieved,
        caveats=['Local timing extrapolation, not a proven optimum; module and trajectory costs can change.',
                 'A larger enumeration quota repeats earlier prefixes; this API has no resume cursor.',
                 'Keep the supplied grammar; inspect structural groups before choosing expansion or coefficient refinement.'])


def run(public, output, ast_budget=65536, banks=768, search_seed=0, seconds=90,
        target_mse=None, host='rack1', notify=True):
    started=time.time();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        temp=output/(name+'.tmp');game.write_json(temp,value);temp.replace(output/name)
    save('intent.json',dict(started_at=started,source=str(public),kind='bounded_same_grammar_feedback'))
    with game.public_input(public) as directory:
        config,problem,states,rows,splits=game.load_challenge(directory)
    request,coverage=prepare(config,ast_budget,banks,search_seed,seconds,target_mse)
    save('grammar.json',config);save('coverage.json',coverage)
    from client import Client, TERMINAL
    from telegram_notify import send, terminal_message
    def notification(message,name):
        if not notify:return
        try:save(name,dict(message=message,message_id=send(message),sent_at=time.time()))
        except RuntimeError as error:save(name,dict(error=str(error)))
    notification(f'Odezza START — bounded search of the supplied grammar: {coverage["requested_asts"]:,} ASTs × {banks} rows. Original depth and operators preserved.','telegram-start.json')
    with Client(host=host,notify=False) as client:
        before=time.time()
        datasets={role:client.upload(states,[dict(trajectory_id=i,**rows[i]) for i in ids]) for role,ids in splits.items()}
        pid=client.problem(dict(problem,data=datasets));save('registration.json',dict(problem_id=pid,datasets=datasets))
        request.update(problem_id=pid,request_id='game-probe-'+game.digest([pid,request,time.time_ns()])[:24])
        save('request.json',request);submission=time.time();state=client.start(request);save('job.json',state)
        try:
            deadline=time.monotonic()+seconds+30
            while True:
                state=client.status(state['campaign_id']);save('latest.json',state)
                if state['status'] in TERMINAL or state['status']=='awaiting_revision':break
                if time.monotonic()>deadline:raise TimeoutError('Probe exceeded its wall budget')
                time.sleep(.25)
        finally:
            if state['status'] not in TERMINAL:
                save('closed-campaign.json',client.cancel(state['campaign_id']))
        observed=time.time()
        timing=dict(input_to_report_seconds=observed-started,registration_seconds=submission-before,
                    submit_to_report_seconds=observed-submission,service_elapsed_seconds=state.get('elapsed_seconds'))
        save('timing.json',timing)
        report=summarize(state,states);save('grouped-report.json',report)
        next_step=recommend(state,request,coverage);save('recommendation.json',next_step)
        if 'suggested_ast_budget' in next_step:
            next_request,next_coverage=prepare(config,next_step['suggested_ast_budget'],banks,search_seed,seconds,target_mse)
            next_request.update(problem_id=pid,request_id=request['request_id']+'-next')
            save('proposed-next-request.json',next_request);save('proposed-next-coverage.json',next_coverage)
        if state['status']=='verified':notification(terminal_message(state,'Same-grammar probe',observed-started),'telegram-result.json')
        else:notification('Odezza first report ready — '+state['status']+f'. Elapsed {observed-started:.2f} s. Grouped results and a next-request recommendation saved; no second search submitted.','telegram-result.json')
        return dict(status=state['status'],**timing,report=str(output/'grouped-report.json'),recommendation=next_step)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('public',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--asts',type=int,default=65536);p.add_argument('--banks',type=int,default=768)
    p.add_argument('--search-seed',type=int,default=0);p.add_argument('--seconds',type=float,default=90)
    p.add_argument('--target-mse',type=float);p.add_argument('--host',default='rack1');p.add_argument('--no-notify',action='store_true')
    a=p.parse_args()
    try:result=run(a.public,a.out,a.asts,a.banks,a.search_seed,a.seconds,a.target_mse,a.host,not a.no_notify)
    except (ValueError,RuntimeError,OSError) as error:p.exit(2,str(error)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
