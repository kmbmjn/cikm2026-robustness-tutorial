"""Provenance-rich reporting; participant writes are restricted to resources/outputs/.

M6 evaluates a repair, decides Go or Hold, repeats a fixed quality check, and
verifies recovery after rollback. Average quality loss and severe query losses
feed one quality alert; ranking stability stays in an optional appendix.
Tolerances change decisions, never a measurement or the evidence hash.
"""
from datetime import datetime, timezone
from importlib.metadata import version
import json
from numbers import Real
from pathlib import Path
import platform
import numpy as np
from .config import ROOT, ENCODER, RERANKER, fingerprint, sha256, ValidationError
from .retrieval import retrieve
from .metrics import evaluate_suite, paired_bootstrap, checked_ranking
from .interventions import build_vocabulary, input_normalization, select_adapter, vocab_spell_correction
from .perturbations import TOKEN, make_query_typos, perturbation_accounting
from .presentation import interpretation_html, show_interpretation

def json_safe(value):
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value,np.bool_): return bool(value)
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer): return int(value)
    return value

def intervention_comparison(ctx,severity=1,selection=None):
    """Reference comparison written by cache preparation; M6 passes an audited selection explicitly."""
    import pandas as pd
    selection=select_adapter(ctx) if selection is None else selection
    rows=[]; evaluations={}
    for name in ["none","input_normalization","vocab_spell_correction","repair_adapter"]:
        kwargs={"intervention":name,"adapter":selection["adapter"]}
        clean=retrieve(ctx,severity=0,**kwargs)
        shifted=retrieve(ctx,severity=severity,**kwargs)
        ev=evaluate_suite(ctx,clean,shifted); evaluations[name]=ev
        s=ev["summary"]
        timed = _timing_matches(ctx, name, severity, selection)
        rows.append({"intervention":name,"clean_ndcg":s["clean_ndcg"],"shifted_ndcg":s["shifted_ndcg"],
                     "bottom_shifted_ndcg":s["bottom_shifted_ndcg"],"worst_paired_delta":s["worst_paired_delta"],
                     "rbo":s["rbo"],"historical_ms/query":ctx.timings["methods"][name]["ms_per_query"] if timed else np.nan,
                     "test_queries":s["n_evaluated"],"reranker":"off (matched dense-only comparison)"})
    return pd.DataFrame(rows).set_index("intervention"),evaluations

INTERVENTIONS = ('none', 'input_normalization', 'vocab_spell_correction', 'repair_adapter')
REPAIRS = INTERVENTIONS[1:]
LABELS = {'none': 'Original search', 'input_normalization': 'Input normalization',
          'vocab_spell_correction': 'Vocabulary correction', 'repair_adapter': 'Repair adapter'}
# Example policy defaults. A team fixes its own values before
# evaluation, from normal run-to-run variation and its service requirements.
POLICY = {'clean_drop_tolerance': 0.01, 'quality_tolerance': 0.01}
TOL = 1e-12  # Differences this small count as equal.
SEVERE_NDCG_DROP = 0.5
SEVERE_RULE = ('a query loses every relevant paper from the top 10, or its nDCG@10 falls by at least '
               f'{SEVERE_NDCG_DROP:g}, compared with the reference on the same input')
QUALITY = {'clean_ndcg': 'nDCG@10, original queries', 'typo_ndcg': 'nDCG@10, typo queries',
           'clean_recall': 'Recall@10, original queries', 'typo_recall': 'Recall@10, typo queries'}
SYNTHETIC_REGRESSION = {
    'name': 'Simulated faulty typo-handling update',
    'story': ("A query-handling bug mixes up saved results. It gives 10% of typo queries the results for another "
              'typo query. Original queries skip this step and keep their results.'),
    'affected_fraction': 0.1, 'seed': 42,
    'selection': ('misspelled test queries ordered by fingerprint([seed, "synthetic_regression", query_id]); '
                  "each affected query receives the next affected query's typo results, cyclically in test order"),
    'scope': 'simulated result-list changes in a copy of the fixed quality-check results'}
RESPONSE_PLAN = [
    {'step': 'Inspect', 'action': 'Inspect a failed query and its relevant papers; check the faulty query-handling update'},
    {'step': 'Roll back', 'action': 'Restore the system that produced the saved reference'},
    {'step': 'Recheck', 'action': 'Rerun the same quality check; confirm recovery and clear the alert'}]
NEXT_STEPS = ['Improve the repairs on hold, then repeat the same release checks before use.',
              'Test with real user typos. Set tolerances using the score variation seen in normal runs.',
              'Check the answers an LLM writes from the retrieved papers.']


def _validate_settings(ctx, intervention, severity, policy):
    if ctx.mode != 'cached':
        raise ValidationError('M6 reports use verified cached CPU evidence only; live/toy reports are not supported.')
    if intervention not in INTERVENTIONS:
        raise ValidationError('Choose none, input_normalization, vocab_spell_correction, or repair_adapter.')
    if type(severity) is not int or severity not in range(4):
        raise ValidationError('Report severity must be an integer from 0 to 3.')
    for name, value in policy.items():
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or not 0 <= value <= 1):
            raise ValidationError(f'{name} must be a finite number in [0,1].')


def select_report_adapter(ctx, lambdas=(.1, 1., 10., 100.), train_noise_severity=1,
                          seed=42, validation_severity=1):
    """M6 fits and selects on M4's text-only split audit; source splits stay unchanged."""
    from .improvement import split_audit
    audit = split_audit(ctx)
    return select_adapter(ctx, lambdas=lambdas, train_noise_severity=train_noise_severity,
                          seed=seed, validation_severity=validation_severity,
                          train_ids=audit['ids']['train'], validation_ids=audit['ids']['validation'])


def _verified_selection(ctx, selection):
    """A caller-supplied matrix is not evidence that train/validation isolation held."""
    if selection is None:
        return select_report_adapter(ctx)  # Fit/validation stress stays 1 even at test severity 0.
    required = {'adapter', 'lambda', 'lambda_grid', 'train_noise_severity', 'seed', 'validation_severity',
                'fit_query_ids', 'validation_query_ids'}
    if not isinstance(selection, dict) or not required <= selection.keys():
        raise ValidationError('Report selection needs the complete select_adapter result, including its validation config.')
    verified = select_report_adapter(ctx, lambdas=selection['lambda_grid'],
                              train_noise_severity=selection['train_noise_severity'], seed=selection['seed'],
                              validation_severity=selection['validation_severity'])
    if (list(selection['fit_query_ids']) != verified['fit_query_ids'] or
            list(selection['validation_query_ids']) != verified['validation_query_ids']):
        raise ValidationError('Report selection must use the audited train-only/validation-only IDs. '
                              'Use select_report_adapter(ctx); original ID-only fits contain duplicate query texts.')
    if (selection['lambda'] != verified['lambda'] or
            not np.array_equal(np.asarray(selection['adapter']), verified['adapter'])):
        raise ValidationError('Supplied adapter does not match a train-only fit and validation-only selection.')
    return verified


def rank_change(per_query):
    """Average top-10 rank change, 1 − RBO@10, between original and typo rankings; needs no labels."""
    values = np.asarray(per_query['rbo'], dtype=float)
    if not len(values) or not np.isfinite(values).all() or np.any((values < -TOL) | (values > 1+TOL)):
        raise ValidationError('Rank drift needs finite per-query RBO values in [0,1].')
    drops = np.clip(1-values, 0, 1)
    drops[drops <= TOL] = 0  # Identical lists must not alert at tolerance zero.
    return float(drops.mean())


def canary_summary(per_query):
    """Mean measurements of one canary run over its fixed queries."""
    return {'query_count': int(len(per_query)),
            'clean_ndcg': float(per_query['clean_ndcg'].mean()), 'typo_ndcg': float(per_query['shifted_ndcg'].mean()),
            'clean_recall': float(per_query['clean_recall'].mean()),
            'typo_recall': float(per_query['shifted_recall'].mean()), 'rank_change': rank_change(per_query)}


def severe_failures(reference, current, inputs=('clean', 'shifted')):
    """Queries that lose every relevant paper from the top 10, or fall by ≥ 0.5 nDCG@10, on the same input.

    Largest drop first; ties prefer the original input, then test order.
    """
    if not reference.index.equals(current.index):
        raise ValidationError('Compare runs on the same query IDs, in the same order.')
    order = {q: i for i, q in enumerate(reference.index)}
    rows = []
    for condition in inputs:
        before, after = reference[f'{condition}_ndcg'], current[f'{condition}_ndcg']
        lost = (reference[f'{condition}_recall'] > TOL) & (current[f'{condition}_recall'] <= TOL)
        for q in reference.index[(before-after >= SEVERE_NDCG_DROP-TOL) | lost]:
            rows.append({'query_id': q, 'input': 'original' if condition == 'clean' else 'typo',
                         'reference_ndcg': float(before[q]), 'current_ndcg': float(after[q]),
                         'reference_recall': float(reference.loc[q, f'{condition}_recall']),
                         'current_recall': float(current.loc[q, f'{condition}_recall']),
                         'lost_all_relevant': bool(lost[q])})
    return sorted(rows, key=lambda r: (r['current_ndcg']-r['reference_ndcg'], r['input'] != 'original',
                                       order[r['query_id']]))


def compare_with_reference(reference, current):
    """Policy-free differences between one canary run and the approved reference."""
    ref, cur = canary_summary(reference), canary_summary(current)
    return {'reference': ref, 'current': cur, 'additional_drift': cur['rank_change']-ref['rank_change'],
            'quality_change': {key: cur[key]-ref[key] for key in QUALITY},
            'severe_failures': severe_failures(reference, current)}


def alert_status(comparison, policy):
    """Combine average quality loss and severe query losses into one quality alert."""
    change = comparison['quality_change']
    worst = min(QUALITY, key=lambda key: change[key])  # The first listed measure wins a tie.
    count = len(comparison['severe_failures'])
    alerts = {
        'quality_regression': {'value': change[worst], 'measure': QUALITY[worst],
                               'tolerance': policy['quality_tolerance'],
                               'alert': -change[worst] > policy['quality_tolerance']+TOL,
                               'rule': (f"average nDCG@10 or Recall@10 falls more than {policy['quality_tolerance']:g} "
                                        'below the saved score'), 'owner': 'Search Team'},
        'severe_failure': {'value': count, 'alert': count > 0, 'owner': 'Search Team',
                           'rule': 'any query loses all relevant papers from the top 10 or falls 0.5 or more in nDCG@10'}}
    for alert in alerts.values():
        alert['status'] = 'Alert' if alert['alert'] else 'No alert'
    fired = any(alert['alert'] for alert in alerts.values())
    return {**alerts, 'alert': fired, 'status': 'Alert' if fired else 'No alert'}


def _ci(ci):
    return f"{ci['estimate']:+.4f} [{ci['low']:+.4f}, {ci['high']:+.4f}]"


def _relevant_ranks(ctx, q, ranking):
    positives = {d for d, grade in ctx.qrels[q].items() if grade > 0}
    return [rank for rank, (d, _) in enumerate(ranking, 1) if d in positives]


def _gate_evidence(ctx, evaluations, name, selection, vocabulary):
    """Policy-free deployment evidence: paired gains over original search and broken original queries."""
    base, runs = evaluations['none']['per_query'], evaluations[name]['per_query']
    gain = lambda column: paired_bootstrap(base[column], runs[column], ctx.seed, 500)
    broken = severe_failures(base, runs, inputs=('clean',))  # Reference here: original search.
    if broken:
        ids = [case['query_id'] for case in broken]
        original = retrieve(ctx, ids)
        repaired = retrieve(ctx, ids, intervention=name, adapter=selection['adapter'])
        for case in broken:
            q = case['query_id']; text = ctx.queries[q]
            output = (vocab_spell_correction(text, vocabulary) if name == 'vocab_spell_correction' else
                      input_normalization(text) if name == 'input_normalization' else text)
            words = TOKEN.findall(text), TOKEN.findall(output)
            case.update({'query_text': text, 'method_text': output,
                         'changed_words': ([[a, b] for a, b in zip(*words) if a != b]
                                           if name == 'vocab_spell_correction' else []),
                         'original_search_relevant_ranks': _relevant_ranks(ctx, q, original[q]),
                         'method_relevant_ranks': _relevant_ranks(ctx, q, repaired[q]),
                         'method_first_paper': ctx.corpus[repaired[q][0][0]].split('\n')[0]})
    return {'method': name, 'comparison': 'method minus original search on the same queries and inputs',
            'typo_gain': gain('shifted_ndcg'), 'clean_change': gain('clean_ndcg'),
            'typo_recall_gain': gain('shifted_recall'), 'clean_recall_change': gain('clean_recall'),
            'broken_original_queries': broken}


def _broken_text(broken):
    case, n = broken[0], len(broken)
    words = ', '.join(f'{a} → {b}' for a, b in case.get('changed_words', []))
    detail = (f"the claim “{case['query_text']}” (nDCG@10 {case['reference_ndcg']:.4f} → {case['current_ndcg']:.4f}"
              + (f', where correction replaces {words})' if words else ')'))
    return f'it causes a severe loss on 1 original query: {detail}' if n == 1 else f'it causes severe losses on {n} original queries, for example {detail}'


def _decision_reason(evidence, checks, tolerance):
    label, gain = LABELS[evidence['method']], evidence['typo_gain']
    gain_ok, cost_ok, broken_ok = (check['passed'] for check in checks)
    if gain_ok and cost_ok and broken_ok:
        return (f'{label}: typo nDCG@10 improves by {_ci(gain)} over original search. '
                f'Original-query loss stays within {tolerance:g}, and no original query has a severe loss. '
                'Use this repair in the simulation and monitor it against its saved scores.')
    other = []
    if not cost_ok:
        other.append(f"its original-query nDCG@10 gain interval, {_ci(evidence['clean_change'])}, "
                     f"allows a loss larger than the {tolerance:g} tolerance")
    if not broken_ok:
        other.append(_broken_text(evidence['broken_original_queries']))
    if gain_ok:
        finding = f'it improves typo queries ({_ci(gain)}). However, ' + '. Also, '.join(other)
    else:
        finding = (f'it gives the same after-typo scores as original search ({_ci(gain)}), so there is no gain to deploy'
                   if max(abs(gain['low']), abs(gain['high'])) <= TOL else
                   f'it lowers after-typo nDCG@10 compared with original search ({_ci(gain)})' if gain['high'] < -TOL else
                   f'its typo-gain interval is {_ci(gain)}. It includes zero, so we cannot conclude that the repair helps')
        finding += f". Also, {'. Also, '.join(other)}" if other else ''
    return (f'{label}: {finding}. Keep original search in use. Improve the repair '
            'and test it again before use.')


def deployment_decision(evidence, policy):
    """Educational Go/Hold gate over stored evidence: Go needs all three checks."""
    gain, cost, broken = evidence['typo_gain'], evidence['clean_change'], evidence['broken_original_queries']
    tolerance = policy['clean_drop_tolerance']
    checks = [
        {'check': 'Typo gain', 'rule': 'lower 95% bound of the after-typo nDCG@10 gain over original search > 0',
         'value': gain, 'passed': bool(gain['low'] > TOL)},
        {'check': 'Original-query cost',
         'rule': f'lower 95% bound of the original-query nDCG@10 gain over original search ≥ −{tolerance:g}',
         'value': cost, 'passed': bool(cost['low'] >= -tolerance-TOL)},
        {'check': 'Broken original queries', 'rule': 'none: ' + SEVERE_RULE.replace('the reference', 'original search'),
         'value': len(broken), 'passed': not broken}]
    status = 'Go' if all(check['passed'] for check in checks) else 'Hold'
    return {'method': evidence['method'], 'status': status, 'checks': checks,
            'reason': _decision_reason(evidence, checks, tolerance)}


def _canary_run(ctx, system, severity, selection):
    """One canary run: the fixed test queries, original and with their fixed typos."""
    kwargs = {'intervention': system, 'adapter': selection['adapter']}
    return {'clean': retrieve(ctx, severity=0, **kwargs), 'typo': retrieve(ctx, severity=severity, **kwargs)}


def synthetic_regression(ctx, runs, severity):
    """Simulated faulty typo-handling update, applied to an in-memory copy of one canary run."""
    seed = SYNTHETIC_REGRESSION['seed']
    misspelled = [q for q in ctx.test_ids if severity and ctx.variants[str(severity)][q]['edit_count']]
    count = round(SYNTHETIC_REGRESSION['affected_fraction']*len(misspelled))
    chosen = set(sorted(misspelled, key=lambda q: fingerprint([seed, 'synthetic_regression', q]))[:count])
    affected = [q for q in ctx.test_ids if q in chosen] if count > 1 else []
    typo = dict(runs['typo'])
    for q, source in zip(affected, affected[1:]+affected[:1]):
        typo[q] = list(runs['typo'][source])
    return {'clean': dict(runs['clean']), 'typo': typo}, affected


def _describe_failures(ctx, cases, reference_runs, runs, severity):
    for case in cases:
        q = case['query_id']; key = 'clean' if case['input'] == 'original' else 'typo'
        case.update({'query_text': ctx.queries[q] if key == 'clean' else ctx.variants[str(severity)][q]['text'],
                     'reference_relevant_ranks': _relevant_ranks(ctx, q, reference_runs[key][q]),
                     'current_relevant_ranks': _relevant_ranks(ctx, q, runs[key][q]),
                     'current_first_paper': ctx.corpus[runs[key][q][0][0]].split('\n')[0]})


def _diagnosis(simulated):
    changed = simulated['changed_rankings']
    if not changed['original'] and not changed['typo']:
        return {'cause': None, 'owner': None, 'finding': 'Every test query kept the same ordered top 10, so there is nothing to diagnose.'}
    if not changed['original']:
        return {'cause': 'typo handling', 'owner': 'Search Team',
                'finding': ('Original queries kept their top ten. Only typo results changed. '
                            'Check the query-handling update first.')}
    return {'cause': 'index, documents, or encoder', 'owner': 'Search Team',
            'finding': 'Original queries also received different result lists, so check the index, the documents, and the encoder.'}


def _monitoring(ctx, system, severity, selection):
    """Policy-free canary measurements of the approved system: reference, rerun, simulation, rollback."""
    reference_runs = _canary_run(ctx, system, severity, selection)  # Recorded when the system was approved.
    current_runs = _canary_run(ctx, system, severity, selection)  # Scheduled rerun; nothing has changed.
    faulty_runs, affected = synthetic_regression(ctx, current_runs, severity)
    rollback_runs = _canary_run(ctx, system, severity, selection)  # Rerun after rolling back the faulty update.
    frame = lambda runs: evaluate_suite(ctx, runs['clean'], runs['typo'])['per_query']
    reference, results = frame(reference_runs), {}
    for name, runs in [('observed', current_runs), ('simulated_regression', faulty_runs),
                       ('after_rollback', rollback_runs)]:
        comparison = compare_with_reference(reference, frame(runs))
        comparison['changed_rankings'] = {
            label: sum(checked_ranking(runs[key][q])[:10] != checked_ranking(reference_runs[key][q])[:10]
                       for q in ctx.test_ids) for label, key in [('original', 'clean'), ('typo', 'typo')]}
        _describe_failures(ctx, comparison['severe_failures'], reference_runs, runs, severity)
        results[name] = comparison
    simulated, verified = results['simulated_regression'], results['after_rollback']
    simulated.update({**SYNTHETIC_REGRESSION, 'affected_query_ids': affected,
                      'misspelled_count': sum(bool(severity and ctx.variants[str(severity)][q]['edit_count'])
                                              for q in ctx.test_ids)})
    failed = list(dict.fromkeys(case['query_id'] for case in simulated['severe_failures']))
    remaining = {case['query_id'] for case in verified['severe_failures']}
    verified['recovered_query_ids'] = [q for q in failed if q not in remaining]
    return {'scope': 'Offline quality-check simulation using fixed saved SciFact results.',
            'canary_set': {'query_count': len(ctx.test_ids), 'severity': severity, 'typo_seed': ctx.seed,
                           'description': ('official SciFact test queries with known relevant papers, each run '
                                           'original and with its fixed synthetic typos')},
            'monitored_system': system, 'monitored_label': LABELS[system], 'reference': results['observed']['reference'],
            'reference_definition': 'quality check of the system in use saved before the update; same queries, typos, corpus and top 10',
            'severe_failure_rule': SEVERE_RULE, 'observed': results['observed'],
            'simulated_regression': simulated, 'after_rollback': verified}


def build_robustness_report(ctx, metrics=None, intervention='repair_adapter', severity=1,
                            clean_drop_tolerance=POLICY['clean_drop_tolerance'],
                            quality_tolerance=POLICY['quality_tolerance'], selection=None):
    """Reconstruct a verified full-test report, then apply the deployment and alert policy.

    This report's fixed contract is dense-only, top-20, K=10, p=.9, tail=.1,
    500 paired bootstrap resamples. A mismatched external evaluation must not
    acquire the reference dataset/system provenance by being passed to this API.
    Tolerances change decisions only; measurements and the evidence and
    monitoring hashes stay the same.
    """
    policy = {'clean_drop_tolerance': clean_drop_tolerance, 'quality_tolerance': quality_tolerance}
    _validate_settings(ctx, intervention, severity, policy)
    policy = {key: float(value) for key, value in policy.items()}
    from .improvement import split_audit
    audit = split_audit(ctx)
    selection = _verified_selection(ctx, selection)
    table, evaluations = intervention_comparison(ctx, severity, selection)
    measured = evaluations[intervention]
    if metrics is not None:
        try:
            import pandas as pd
            pd.testing.assert_frame_equal(metrics['per_query'], measured['per_query'], check_exact=True)
            for key in ['summary', 'ci', 'query_ids', 'label_valid']:
                if json_safe(metrics[key]) != json_safe(measured[key]):
                    raise AssertionError(key)
        except (KeyError, AssertionError, TypeError, ValueError) as error:
            raise ValidationError('Supplied metrics do not match the declared full-test cached run. '
                                  'Use metrics=None to reconstruct verified evidence.') from error
    s = measured['summary']; baseline = evaluations['none']; sf = measured['per_query']; bf = baseline['per_query']
    selection_config = {k: selection[k] for k in ['lambda', 'lambda_grid', 'train_noise_severity', 'seed', 'validation_severity']}
    reference_selection = (selection['train_noise_severity'] == selection['validation_severity'] == 1
                           and selection['seed'] == 42
                           and tuple(selection['lambda_grid']) == (0.1, 1., 10., 100.))
    is_reference = intervention == 'repair_adapter' and severity == 1 and policy == POLICY and reference_selection
    source = 'cached reference run' if is_reference else 'configured cached run'
    config = {'intervention': intervention, 'severity': severity, 'k': 10, 'p': .9, 'tail_fraction': .1,
              'n_boot': 500, 'bootstrap_seed': ctx.seed, 'policy': policy,
              'severe_failure_rule': SEVERE_RULE, 'synthetic_regression': SYNTHETIC_REGRESSION,
              'adapter_selection': selection_config, 'query_ids': ctx.test_ids,
              'split_audit_hash': fingerprint(audit),
              'fit_query_ids_hash': fingerprint(selection['fit_query_ids']),
              'validation_query_ids_hash': fingerprint(selection['validation_query_ids'])}
    variants = ({q: ctx.variants[str(severity)][q] for q in ctx.test_ids} if severity else
                make_query_typos({q: ctx.queries[q] for q in ctx.test_ids}, severity=0, seed=ctx.seed))
    accounting = perturbation_accounting(variants)
    accounting['actual_edit_count_histogram'] = {str(n): sum(v['edit_count'] == n for v in variants.values()) for n in range(4)}
    accounting['retention_policy'] = 'Keep every requested test query, including queries with fewer actual edits than the selected maximum.'
    accounting['generation_exclusions'] = {q: v['excluded_reason'] for q, v in variants.items() if v['excluded_reason']}
    comparisons = {'baseline': 'none', 'baseline_mean_and_tail': baseline['summary'],
                   'clean_difference': paired_bootstrap(bf.clean_ndcg, sf.clean_ndcg, ctx.seed, 500),
                   'shifted_difference': paired_bootstrap(bf.shifted_ndcg, sf.shifted_ndcg, ctx.seed, 500),
                   'sign': 'selected intervention minus no-intervention dense baseline; positive favors selected',
                   'comparison_scope': 'same test IDs, qrels, corruption instances, full corpus and 20 candidates per search; compute costs differ'}
    vocabulary = build_vocabulary(ctx.corpus)
    gains = {name: _gate_evidence(ctx, evaluations, name, selection, vocabulary) for name in REPAIRS}
    evaluation = {'metrics': ['nDCG@10', 'Recall@10', 'normalized finite RBO@10', 'common-item Kendall tau'],
                  'mean_and_tail': s, 'paired_ci': measured['ci'],
                  'paired_ci_definition': 'shifted minus clean within the selected system; percentile 95%, 500 query-pair resamples, seed 42',
                  'baseline_comparison': comparisons,
                  'all_interventions': table.reset_index().to_dict(orient='records'),
                  'performance': [{'intervention': name, 'clean_ndcg': ev['summary']['clean_ndcg'],
                                   'typo_ndcg': ev['summary']['shifted_ndcg'], 'clean_recall': ev['summary']['clean_recall'],
                                   'typo_recall': ev['summary']['shifted_recall'], 'rank_change': rank_change(ev['per_query'])}
                                  for name, ev in evaluations.items()],
                  'intervention_gains': gains,
                  'intervention_gain_definition': ('method minus original search on the same query and input; '
                                                   'percentile 95%, 500 query-pair resamples, seed 42'),
                  'ranking_stability': {k: s[k] for k in ['rbo', 'tau', 'common_count', 'coverage']},
                  'tau_defined_queries': int(sf.tau.notna().sum()),
                  'judged_coverage': s['judged_coverage'],
                  'per_query': sf.reset_index().to_dict(orient='records'),
                  'metric_contract': {'ndcg': 'linear gains; log2(rank+1) discount; IDCG from all positive qrels',
                      'recall': 'fraction of original positive-qrel documents in top-10',
                      'unjudged': 'zero evaluation gain, not known irrelevant; judged coverage is judgments/top-10, not trust',
                      'zero_idcg': 'exclude undefined relevance scores and report count',
                      'rbo': 'sum_(d=1..10) (1-p)*p^(d-1)*overlap@d / (1-p^10); not extrapolated RBO',
                      'kendall': 'tau only on shared top-10 IDs; undefined below 2 shared IDs; coverage=shared/10',
                      'tails': 'separately select ceil(.1*n) lowest shifted nDCG and lowest shifted-clean differences',
                      'uncertainty': 'conditional on fixed model/corruptions and exchangeable query pairs; no across-domain or seed-variance guarantee'}}
    decisions = {name: deployment_decision(gains[name], policy) for name in REPAIRS}
    decision = ({'method': 'none', 'status': 'Hold', 'checks': [],
                 'reason': 'Original search: no new method is proposed, so original search stays in use.'}
                if intervention == 'none' else decisions[intervention])
    approved = intervention if decision['status'] == 'Go' else 'none'
    monitoring = _monitoring(ctx, approved, severity, selection)
    alerts = {name: alert_status(monitoring[name], policy) for name in ('observed', 'simulated_regression', 'after_rollback')}
    verified = monitoring['after_rollback']
    code_paths = sorted((ROOT/'tutorial_utils').glob('*.py')) + [ROOT/'requirements.txt']
    current_files = {str(p.relative_to(ROOT)): sha256(p) for p in code_paths}
    evidence_config = {k: v for k, v in config.items() if k != 'policy'}
    evidence_identity = {'config': evidence_config, 'evaluation': evaluation,
                         'manifest_sha256': sha256(ROOT/'assets/caches/manifest.json'),
                         'current_report_code_fingerprint': fingerprint(current_files)}
    evidence_hash = fingerprint(json_safe(evidence_identity))
    selection_record = {**selection_config, 'selection': selection['selection'],
                        'tie_break': 'validation shifted nDCG descending, clean nDCG descending, lambda ascending',
                        'applied_to_selected_system': intervention == 'repair_adapter',
                        'scope': 'adapter arm only; input methods have no fitted adapter',
                        'fit_query_ids': selection['fit_query_ids'],
                        'selection_query_ids': selection['validation_query_ids'],
                        'validation_scores': selection['validation'].to_dict(orient='records'),
                        'matrix_hash': fingerprint(selection['adapter'].tolist())}
    report = {'schema_version': 6,
        'tutorial': 'Beyond Benchmarks: Robustness for Real-World Information Systems', 'source': source,
        'scope': 'Offline simulation of deployment decisions and monitoring using fixed cached SciFact results.',
        'settings': {'intervention': intervention, 'candidate': LABELS[intervention], 'severity': severity,
                     'policy': policy,
                     'policy_note': ('Deployment and alert tolerances are fixed before evaluation, based on '
                                     'normal run-to-run variation and service quality requirements.')},
        'dataset': {**ctx.manifest['dataset'], 'evaluated_query_count': s['n_evaluated'],
                    'split': 'official BEIR test', 'query_ids': ctx.test_ids,
                    'split_counts': {k: len(v) for k, v in ctx.manifest['splits'].items()},
                    'derived_split_hash': fingerprint(ctx.manifest['splits']),
                    'split_counts_scope': 'original source/cache splits; effective experiment splits below',
                    'effective_split_counts': {k: len(v) for k, v in audit['ids'].items()},
                    'split_audit': audit},
        'system': {'retriever': ENCODER, 'reranker': RERANKER, 'reranker_applied': False,
                   'retrieval_k': 20, 'rerank_k': 0, 'pipeline_reranker_candidate_budget': 20,
                   'preprocessing': ctx.manifest['preprocessing'], 'device': 'cpu', 'dtype': 'float32',
                   'scope': 'matched dense-only comparison; BM25 and dense-to-reranker results from M3 are separate experiments',
                   'model_training': 'frozen broad sentence-pair encoder and MS MARCO reranker; no SciFact encoder fine-tuning; pretraining overlap not audited'},
        'threat_model': {'surface': 'query', 'allowed_perturbations': 'seeded adjacent-keyboard substitution/insertion/deletion in eligible words',
                         'attack_budget': severity, 'budget_unit': 'at most one character edit per eligible word; up to severity words',
                         'protocol': 'fixed synthetic corruption, not adaptive attack or natural typo traffic',
                         'semantic_validity': 'Original SciFact relevance judgments are reused for all query variants.',
                         'accounting': accounting, 'variants_hash': fingerprint(variants)},
        'evaluation': evaluation,
        'deployment_decision': {**decision, 'candidate': intervention, 'candidate_label': LABELS[intervention],
                                'approved_system': approved, 'approved_label': LABELS[approved],
                                'gate': 'Go needs every check; otherwise Hold and keep original search in use.',
                                'policy': {'clean_drop_tolerance': policy['clean_drop_tolerance'],
                                           'broken_query_rule': SEVERE_RULE.replace('the reference', 'original search')},
                                'by_method': decisions},
        'monitoring': monitoring,
        'alerts': {'policy': {'quality_tolerance': policy['quality_tolerance']},
                   'severe_failure_rule': SEVERE_RULE, **alerts},
        'response': {'owner': 'Search Team', 'plan': RESPONSE_PLAN, 'triggered_by': 'simulated regression',
                     'diagnosis': _diagnosis(monitoring['simulated_regression']),
                     'mitigation': {'owner': 'Search Team',
                                    'action': 'Roll back the faulty query-handling update to the system that produced the saved reference.',
                                    'gate': 'Any new fix must pass the release checks before use.'},
                     'verification': {'rerun': 'the same test queries and fixed typos after rollback',
                                      'status': alerts['after_rollback']['status'],
                                      'matches_reference': bool(not verified['severe_failures'] and all(
                                          abs(v) <= TOL for v in verified['quality_change'].values())),
                                      'recovered_query_ids': verified['recovered_query_ids'],
                                      'remaining_severe_failures': len(verified['severe_failures'])}},
        'intervention': {'name': intervention, 'clean_utility': s['clean_ndcg'], 'shifted_utility': s['shifted_ndcg'],
                         'latency': _latency(ctx, intervention, severity, selection), 'validation_selection': selection_record},
        'failure_examples': _failure_examples(ctx, measured, intervention, severity, selection),
        'reproducibility': {'seed': ctx.seed, 'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                               'packages': {p: version(p) for p in ['numpy', 'scipy', 'pandas', 'matplotlib', 'nbformat', 'threadpoolctl']}},
                           'config': config, 'config_hash': fingerprint(json_safe(config)), 'evidence_hash': evidence_hash,
                           'evidence_hash_definition': 'SHA256 of config excluding policy, evaluation, manifest SHA256, and current report code fingerprint; excludes timestamp, tolerances and decisions',
                           'monitoring_hash': fingerprint(json_safe(monitoring)),
                           'monitoring_hash_definition': 'SHA256 of the canary measurements (reference, rerun, simulated regression, rollback); excludes tolerances and alert decisions',
                           'reference_protocol': 'M6 v3: duplicate-text-purged fit/selection; fixed train/validation stress=1 and seed=42; not the historical ID-only adapter cache',
                           'cache_status': ctx.mode, 'cache_config_hash': ctx.manifest['config_hash'],
                           'model_revisions': ctx.manifest['model_revisions'],
                           'data_hashes': {k: v for k, v in ctx.manifest['files'].items() if k.startswith('data/')},
                           'verified_asset_hashes': ctx.manifest['files'], 'manifest_sha256': sha256(ROOT/'assets/caches/manifest.json'),
                           'cache_generation_code_fingerprint': ctx.manifest['code_fingerprint'],
                           'cache_generation_environment': ctx.manifest['environment'],
                           'current_report_code_fingerprint': fingerprint(current_files), 'current_report_code_files': current_files,
                           'cache_generated_at': ctx.manifest['generated_at'], 'run_timestamp': datetime.now(timezone.utc).isoformat()},
        'next_steps': NEXT_STEPS}
    return json_safe(report)


def _timing_matches(ctx, name, severity, selection):
    # Identical lambda/noise/seed do not identify a model trained on different IDs.
    return severity == 1 and (name != 'repair_adapter' or (
        selection['fit_query_ids'] == ctx.train_ids and all(
            selection[k] == ctx.timings['adapter_config'][k]
            for k in ['lambda', 'seed', 'train_noise_severity'])))


def _latency(ctx, name, severity, selection):
    matches = _timing_matches(ctx, name, severity, selection)
    return {'historical_ms_per_query': ctx.timings['methods'][name]['ms_per_query'] if matches else None,
            'matches_selected_config': matches, 'measurement': ctx.timings['provenance'],
            'statistic': 'median of three batch-average ms/query measurements; not request p95 or a significance test',
            'query_ids': ctx.timings['query_ids'], 'query_count': ctx.timings['query_count'],
            'batch_size': ctx.timings['batch_size'], 'repetitions': ctx.timings['repetitions'],
            'warmup_batches': ctx.timings['warmup_batches'], 'platform': ctx.timings['platform'],
            'torch_threads': ctx.timings['torch_threads'], 'blas_threads': ctx.timings['blas_threads'],
            'replicate_ms_per_query': ctx.timings['methods'][name]['replicate_ms_per_query'] if matches else None,
            'timed_severity': 1,
            'timed_adapter': {**ctx.timings['adapter_config'], 'fit_query_ids_hash': fingerprint(ctx.train_ids),
                              'fit_query_count': len(ctx.train_ids)},
            'timing_query_split': 'original ID-derived validation subset; not the duplicate-purged split',
            'not_measured': 'duplicate-purged adapter inference, clean inference, online serving, fallback cost, and production tail latency'}


def _failure_examples(ctx, evaluation, intervention, severity, selection):
    frame = evaluation['per_query'].reset_index().sort_values(['delta_ndcg', 'query_id'])
    ids = frame.loc[frame.delta_ndcg < -1e-12, 'query_id'].head(2).tolist()
    if not ids:
        return []
    kwargs = {'intervention': intervention, 'adapter': selection['adapter']}
    clean = retrieve(ctx, ids, severity=0, **kwargs)
    shifted = retrieve(ctx, ids, severity=severity, **kwargs)
    examples = []
    for q in ids:
        variant = ctx.variants[str(severity)][q]
        examples.append({'query_id': q, 'selection': 'two worst observed paired nDCG drops; exploratory, not prevalence',
                         'original_query': ctx.queries[q], 'shifted_query': variant['text'],
                         'edits': variant['edits'], 'flags': variant['flags'],
                         'clean_ndcg': float(evaluation['per_query'].loc[q, 'clean_ndcg']),
                         'shifted_ndcg': float(evaluation['per_query'].loc[q, 'shifted_ndcg']),
                         'delta_ndcg': float(evaluation['per_query'].loc[q, 'delta_ndcg']),
                         'clean_top10_ids': checked_ranking(clean[q])[:10],
                         'shifted_top10_ids': checked_ranking(shifted[q])[:10],
                         'positive_qrels': {d: g for d, g in ctx.qrels[q].items() if g > 0}})
    return examples


def _reading(ci):
    """Read a paired 95% interval plainly: above zero is higher, below zero is lower, otherwise the same."""
    return 'higher' if ci['low'] > 0 else 'lower' if ci['high'] < 0 else 'same'


def lesson_summary(report):
    """State the displayed result plainly; intervals and detail stay in the reference."""
    s = report['evaluation']['mean_and_tail']; c = report['evaluation']['baseline_comparison']
    base = c['baseline_mean_and_tail']; name = report['intervention']['name']; method = LABELS[name]
    if report['threat_model']['attack_budget'] == 0:
        result = f"No typos are added, so both columns show the original input. {method} scores {s['clean_ndcg']:.4f}."
        return result + (f" Original search scores {base['clean_ndcg']:.4f}." if name != 'none' else '')
    within = _reading(report['evaluation']['paired_ci'])
    scores = f"{s['clean_ndcg']:.4f} → {s['shifted_ndcg']:.4f}"
    who = 'original search' if name == 'none' else method
    result = {'lower': f"With {who}, typos lower the score from {s['clean_ndcg']:.4f} to {s['shifted_ndcg']:.4f}.",
              'same': f"With {who}, the score stays about the same after typos ({scores}).",
              'higher': f"With {who}, the score rises after typos ({scores})."}[within]
    if name == 'none':
        return result
    gap = c['shifted_difference']; versus = f"({s['shifted_ndcg']:.4f} vs {base['shifted_ndcg']:.4f})"
    if _reading(gap) == 'higher':
        return result + f" After typos, {method} beats original search {versus}."
    if _reading(gap) == 'lower':
        return result + f" After typos, {method} scores below original search {versus}."
    same = 'the same as' if gap['low'] == gap['high'] == 0 else 'about the same as'
    ending = ': the typo problem remains.' if within == 'lower' else '.'
    return result + f" After typos, {method} scores {same} original search {versus}{ending}"


def evaluation_reading(report):
    """Clean performance, typo performance and intervention gain for the candidate, in plain words."""
    s = report['evaluation']['mean_and_tail']; c = report['evaluation']['baseline_comparison']
    base = c['baseline_mean_and_tail']; name = report['intervention']['name']; label = 'the ' + LABELS[name].lower()
    within = report['evaluation']['paired_ci']
    note = (' Severity 0 adds no typos, so the after-typo scores repeat the original queries.'
            if report['threat_model']['attack_budget'] == 0 else '')
    if name == 'none':
        return (f"Original search is the system in use: nDCG@10 {s['clean_ndcg']:.4f} on the original queries "
                f"and {s['shifted_ndcg']:.4f} after typos (ΔnDCG@10, after − before: {_ci(within)}). Set `intervention` to a repair to "
                f"evaluate a candidate.{note}")
    gain = c['shifted_difference']
    meaning = ('The interval is above zero, so the gain is shown.' if gain['low'] > TOL else
               'The interval is below zero, so typo queries get worse.' if gain['high'] < -TOL else
               'Both searches give the same scores.' if max(abs(gain['low']), abs(gain['high'])) <= TOL else
               'The interval includes zero, so we cannot conclude that the repair helps.')
    return '\n'.join([
        f"- Clean performance: on the original queries, {label} scores {s['clean_ndcg']:.4f} and original search "
        f"{base['clean_ndcg']:.4f} (gain {_ci(c['clean_difference'])}).",
        f"- Typo performance: {label}'s mean nDCG@10 goes from {s['clean_ndcg']:.4f} on original queries "
        f"to {s['shifted_ndcg']:.4f} after typos (after − before: {_ci(within)}).",
        f"- Intervention gain after typos: {gain['estimate']:+.4f} over original search ({s['shifted_ndcg']:.4f} vs "
        f"{base['shifted_ndcg']:.4f}), 95% interval [{gain['low']:+.4f}, {gain['high']:+.4f}]. {meaning}{note}"])


def report_summary(report):
    """Decision, measured result and monitoring status in plain words."""
    d = report['deployment_decision']; a = report['alerts']
    return (f"Decision: {d['status']} for {d['candidate_label']}; system in use: {d['approved_label']}.\n\n"
            f"{lesson_summary(report)}\n\n"
            f"Checking {report['monitoring']['monitored_label'].lower()} against its saved scores: "
            f"current run, {a['observed']['status'].lower()}; simulated regression, {a['simulated_regression']['status'].lower()}.")


def performance_table(report, candidate_only=False):
    """Clean and typo performance, optionally limited to the candidate and baseline."""
    import pandas as pd
    candidate = report['intervention']['name']
    return pd.DataFrame([{'Method': LABELS[r['intervention']] + (' (candidate)' if r['intervention'] == candidate else ''),
                          'nDCG@10 original': f"{r['clean_ndcg']:.4f}", 'nDCG@10 after typos': f"{r['typo_ndcg']:.4f}",
                          'Recall@10 original': f"{r['clean_recall']:.4f}",
                          'Recall@10 after typos': f"{r['typo_recall']:.4f}"} for r in report['evaluation']['performance']
                         if not candidate_only or r['intervention'] in {'none', candidate}])


def table_readings(report):
    """Share concise, measurement-derived readings between notebook and saved report."""
    comparison = report['evaluation']['baseline_comparison']
    gain = comparison['shifted_difference']
    clean = comparison['clean_difference']
    decision = report['deployment_decision']
    monitoring = report['monitoring']
    simulated = monitoring['simulated_regression']
    verified = report['response']['verification']
    if gain['low'] > TOL:
        interval = 'the typo-gain interval is above zero'
        confidence = 'The typo-gain interval is above zero, providing evidence of a benefit on these fixed inputs.'
    elif gain['high'] < -TOL:
        interval = 'the typo-gain interval is below zero'
        confidence = 'The typo-gain interval is below zero, showing harm rather than a benefit on these fixed inputs.'
    elif max(abs(gain['low']), abs(gain['high'])) <= TOL:
        interval = 'the typo gain is exactly zero'
        confidence = 'The typo gain and its interval are zero, so this candidate adds no measured benefit.'
    else:
        interval = 'the typo-gain interval includes zero'
        confidence = 'The typo-gain interval includes zero, so we cannot conclude that the repair helps.'
    passed = sum(result['status'] == 'Go' for result in decision['by_method'].values())
    observed = monitoring['observed']
    unchanged = not observed['severe_failures'] and all(abs(value) <= TOL for value in observed['quality_change'].values())
    monitored = ('The current run matches every saved score and adds no severe failures, so no alert fires.'
                 if unchanged else
                 f"The current run has {len(observed['severe_failures'])} severe failures and status "
                 f"{report['alerts']['observed']['status']}.")
    match = 'match' if verified['matches_reference'] else 'do not match'
    return {
        'settings': f"This run evaluates {decision['candidate_label']} at a maximum of "
                    f"{report['threat_model']['attack_budget']} character edits, with fixed documents and decision tolerances.",
        'evaluation': f"Mean nDCG@10 gain: {gain['estimate']:+.4f} after typos and "
                      f"{clean['estimate']:+.4f} on original input. Here {interval}.",
        'confidence': confidence,
        'decision': f"{passed} of {len(decision['by_method'])} repairs pass all three checks; "
                    f"{decision['candidate_label']} receives {decision['status']} and "
                    f"{decision['approved_label']} stays in use.",
        'monitoring': monitored,
        'alerts': f"The bug affects {len(simulated['affected_query_ids'])} typo queries. "
                  f"{len(simulated['severe_failures'])} have severe failures. Status: "
                  f"{report['alerts']['simulated_regression']['status']}.",
        'verification': f"After rollback, {len(verified['recovered_query_ids'])} failed queries recover and "
                        f"{verified['remaining_severe_failures']} severe failures remain. Scores "
                        f"{match} the saved reference. Status: {verified['status']}.",
        'response': 'Inspect the failed evidence, restore the saved system, and rerun the same check before declaring recovery.',
        'ranking': f"Additional rank drift: {simulated['additional_drift']:+.4f} after the bug, "
                   f"{monitoring['after_rollback']['additional_drift']:+.4f} after rollback. "
                   "These values do not control Go/Hold or alerts."
    }


SHORT_LABELS = {'none': 'original search', 'input_normalization': 'normalization',
                'vocab_spell_correction': 'vocabulary correction', 'repair_adapter': 'the adapter'}


def _join(items):
    items = list(items)
    return items[0] if len(items) == 1 else ' and '.join(items) if len(items) == 2 else \
        ', '.join(items[:-1]) + ', and ' + items[-1]


def _hold_reason(name, decision):
    """Plain words for the checks one repair fails."""
    gain_check, cost_check, broken_check = decision['checks']
    gain, problems = gain_check['value'], []
    if not gain_check['passed']:
        problems.append('brings no typo gain' if max(abs(gain['low']), abs(gain['high'])) <= TOL else
                        'makes typo queries worse' if gain['high'] < -TOL else 'shows no clear typo gain')
    if not cost_check['passed']:
        problems.append('costs too much on correctly written queries')
    if not broken_check['passed']:
        count = broken_check['value']
        problems.append(f"badly hurts {count} correctly written quer{'y' if count == 1 else 'ies'}")
    return (f"{SHORT_LABELS[name]} helps typo queries but {_join(problems)}" if gain_check['passed'] else
            f"{SHORT_LABELS[name]} {_join(problems)}")


def takeaways(report):
    """One plain-language 'Interpretation:' sentence for each M6 result, derived from the report."""
    name = report['intervention']['name']; method = LABELS[name]
    severity = report['threat_model']['attack_budget']
    s = report['evaluation']['mean_and_tail']; c = report['evaluation']['baseline_comparison']
    base, gain = c['baseline_mean_and_tail'], c['shifted_difference']
    if name == 'none':
        evaluation = ('No repair is selected and no typos are added, so both columns show the same original-query score.'
                      if severity == 0 else
                      f"No repair is selected, so this shows only what typos cost original search "
                      f"({base['clean_ndcg']:.4f} → {base['shifted_ndcg']:.4f}).")
    elif severity == 0:
        evaluation = (f"Without typos both columns repeat the original queries, so this run shows only how "
                      f"{SHORT_LABELS[name]} changes correctly written queries ({c['clean_difference']['estimate']:+.4f}).")
    elif gain['low'] > TOL:
        evaluation = (f"{method} raises after-typo nDCG@10 from {base['shifted_ndcg']:.4f} to {s['shifted_ndcg']:.4f}, "
                      'and the whole 95% interval of the gain is above zero, so the typo benefit is clear on these queries.')
    elif max(abs(gain['low']), abs(gain['high'])) <= TOL:
        evaluation = f'{method} gives exactly the same scores as original search, so it cannot repair any typo damage.'
    elif gain['high'] < -TOL:
        evaluation = (f"{method} lowers after-typo nDCG@10 from {base['shifted_ndcg']:.4f} to {s['shifted_ndcg']:.4f}, "
                      'so it makes typo queries worse.')
    else:
        evaluation = (f"{method} changes after-typo nDCG@10 from {base['shifted_ndcg']:.4f} to {s['shifted_ndcg']:.4f}, "
                      'but the 95% interval of the gain includes zero, so this test cannot show that it repairs typo damage.')

    d = report['deployment_decision']
    passed = [LABELS[repair] for repair, result in d['by_method'].items() if result['status'] == 'Go']
    reasons = [_hold_reason(repair, result) for repair, result in d['by_method'].items() if result['status'] != 'Go']
    if d['status'] == 'Go':
        decision = (f"{d['candidate_label']} passes all three checks, so it replaces original search in this "
                    'simulation and is monitored from now on.')
    elif not passed:
        decision = 'No repair is ready to use: ' + _join(reasons) + '.'
    else:
        decision = (f"{_join(passed)} pass{'es' if len(passed) == 1 else ''} all three checks, but the candidate does "
                    'not, so original search stays in use: ' + _join(reasons) + '.')

    m = report['monitoring']; observed = m['observed']
    system = m['monitored_label'][:1].lower() + m['monitored_label'][1:]
    unchanged = not observed['severe_failures'] and all(abs(v) <= TOL for v in observed['quality_change'].values())
    monitoring = (f'Rerunning the same check reproduces every saved score, so {system} has not changed and needs no action.'
                  if unchanged else
                  f'The rerun no longer matches the saved scores, so find out what changed in {system} before relying on it.')

    a = report['alerts']; sim = m['simulated_regression']
    if not sim['misspelled_count']:
        alerts = 'With no typos the faulty step never runs, so neither run triggers an alert.'
    else:
        quality, severe = a['simulated_regression']['quality_regression'], a['simulated_regression']['severe_failure']
        tolerance = a['policy']['quality_tolerance']
        quiet = ('The unchanged run passes both checks' if a['observed']['status'] == 'No alert' else
                 'Even the unchanged run triggers an alert')
        count = severe['value']
        typo = 'typo ' if all(case['input'] == 'typo' for case in sim['severe_failures']) else ''
        queries = f"{count} {typo}quer{'y suffers' if count == 1 else 'ies suffer'}"
        measure = quality['measure'].replace(', ', ' on ')
        if quality['alert'] and severe['alert']:
            alerts = (f"{quiet}, while the bug fails both: average {measure} drops by {-quality['value']:.4f}, "
                      f"more than the {tolerance:g} tolerance, and {queries} severe losses, so the alert fires.")
        elif severe['alert']:
            alerts = (f"{quiet}; the bug's average loss stays within the {tolerance:g} tolerance, but {queries} "
                      'severe losses, so the alert still fires.')
        elif quality['alert']:
            alerts = (f"{quiet}, while the bug lowers average {measure} by {-quality['value']:.4f}, more than the "
                      f"{tolerance:g} tolerance, so the alert fires even though no single query fails severely.")
        else:
            alerts = 'Neither check catches the simulated bug at these tolerances, so it would go unnoticed.'

    v = report['response']['verification']
    verification = ('Rolling back restores every saved score and clears the alert, confirming that the faulty '
                    'update caused the loss.' if v['matches_reference'] and v['status'] == 'No alert' else
                    f"After rollback, {v['remaining_severe_failures']} severe failures remain, so recovery is not "
                    'confirmed yet.')

    if not sim['misspelled_count']:
        ranking = 'With no typos, original and typo lists are identical in every run, so there is no drift to compare.'
    else:
        extra, after = sim['additional_drift'], m['after_rollback']['additional_drift']
        change = ('agree less with' if extra > TOL else 'agree more with' if extra < -TOL else 'agree as much with')
        ranking = (f'The bug makes typo results {change} the original results (additional drift {extra:+.4f}), '
                   + ('and rollback returns the drift to its saved level' if abs(after) <= TOL else
                      f'and {after:+.4f} remains after rollback')
                   + '; drift shows instability, not whether evidence was lost.')
    return {'evaluation': evaluation, 'decision': decision, 'monitoring': monitoring, 'alerts': alerts,
            'verification': verification, 'ranking': ranking}


def decision_table(report):
    """Each repair's three gate checks under the current policy."""
    import pandas as pd
    d = report['deployment_decision']; rows = []
    for name, decision in d['by_method'].items():
        gain, cost, broken = decision['checks']
        cases = report['evaluation']['intervention_gains'][name]['broken_original_queries']
        count = str(len(cases))
        mark = lambda check: 'pass' if check['passed'] else 'fail'
        rows.append({'Method': LABELS[name] + (' (candidate)' if name == d['candidate'] else ''),
                     'Typo gain [95% CI]': f"{mark(gain)}: {_ci(gain['value'])}",
                     'Original-query gain [95% CI]': f"{mark(cost)}: {_ci(cost['value'])}",
                     'Severe original-query losses': f'{mark(broken)}: {count}', 'Decision': decision['status']})
    return pd.DataFrame(rows)


def _canary_rows(report, columns):
    m = report['monitoring']; rows = []
    runs = {'Saved reference': m['reference'], 'Current run': m['observed']['current'],
            'Simulated regression': m['simulated_regression']['current'], 'After rollback': m['after_rollback']['current']}
    severe = {'Saved reference': 0, 'Current run': len(m['observed']['severe_failures']),
              'Simulated regression': len(m['simulated_regression']['severe_failures']),
              'After rollback': len(m['after_rollback']['severe_failures'])}
    for key, label in QUALITY.items():
        rows.append({'Quality measure': label, **{c: f'{runs[c][key]:.4f}' for c in columns}})
    rows.append({'Quality measure': 'Severe failures', **{c: str(severe[c]) for c in columns}})
    return rows


def monitoring_table(report):
    """Saved quality scores and the current run, with no ranking-drift diagnostics."""
    import pandas as pd
    rows = _canary_rows(report, ['Saved reference', 'Current run'])
    observed = report['monitoring']['observed']
    changes = [observed['quality_change'][key] for key in QUALITY]
    for row, change in zip(rows, changes):
        row['Current − reference'] = f'{change:+.4f}'
    rows[-1]['Current − reference'] = '–'
    return pd.DataFrame(rows)


def _alert_value(key, alert):
    result = 'fail' if alert['alert'] else 'pass'
    if key == 'severe_failure':
        return f"{alert['value']} quer{'y' if alert['value'] == 1 else 'ies'}, {result}"
    measure = f" ({alert['measure']})" if key == 'quality_regression' and alert['value'] < -TOL else ''
    return f"{alert['value']:+.4f}{measure}, {result}"


def alert_table(report):
    """Two quality checks and their single combined alert, without per-signal owners."""
    import pandas as pd
    a = report['alerts']; rows = []
    for key, label in [('quality_regression', 'Average quality loss'), ('severe_failure', 'Severe query loss')]:
        rows.append({'Check': label, 'Triggers when': a['observed'][key]['rule'],
                     'Unchanged run': _alert_value(key, a['observed'][key]),
                     'Simulated regression': _alert_value(key, a['simulated_regression'][key])})
    rows.append({'Check': 'Quality alert', 'Triggers when': 'either check fails',
                 'Unchanged run': a['observed']['status'],
                 'Simulated regression': a['simulated_regression']['status']})
    return pd.DataFrame(rows)


def verification_table(report):
    """Reference, simulated regression and the rerun after rollback."""
    import pandas as pd
    a = report['alerts']
    rows = _canary_rows(report, ['Saved reference', 'Simulated regression', 'After rollback'])
    rows.append({'Quality measure': 'Quality alert', 'Saved reference': 'No alert',
                 'Simulated regression': a['simulated_regression']['status'], 'After rollback': a['after_rollback']['status']})
    return pd.DataFrame(rows)


def response_plan_table(report):
    import pandas as pd
    return pd.DataFrame(report['response']['plan']).rename(
        columns={'step': 'Step', 'action': 'Action'})


def ranking_stability_table(report):
    """Optional original/typo ranking agreement, never an input to the quality alert."""
    import pandas as pd
    monitoring = report['monitoring']
    reference = monitoring['reference']['rank_change']
    runs = [('Saved reference', monitoring['reference']),
            ('Unchanged run', monitoring['observed']['current']),
            ('Simulated regression', monitoring['simulated_regression']['current']),
            ('After rollback', monitoring['after_rollback']['current'])]
    return pd.DataFrame([{'Run': label, 'Rank drift (1 − RBO@10)': f"{summary['rank_change']:.4f}",
                          'Additional drift': f"{summary['rank_change']-reference:+.4f}"}
                         for label, summary in runs])


def _ranks_text(ranks):
    return ('no relevant paper in the top 20' if not ranks else f'relevant paper at rank {ranks[0]}' if len(ranks) == 1
            else 'relevant papers at ranks ' + ', '.join(map(str, ranks)))


def _example_text(case):
    where = 'after typos' if case['input'] == 'typo' else 'original query'
    return (f"Query ({where}): {case['query_text']} Saved result: "
            f"{_ranks_text(case['reference_relevant_ranks'])}, nDCG@10 {case['reference_ndcg']:.4f}. "
            f"Current result: {_ranks_text(case['current_relevant_ranks'])}, nDCG@10 {case['current_ndcg']:.4f}. "
            f"First paper now: {case['current_first_paper']}")


def report_markdown(report):
    """Readable report: settings, decision, performance, monitoring, alerts, failures, response."""
    import pandas as pd
    s = report['evaluation']['mean_and_tail']; c = report['evaluation']['baseline_comparison']
    d = report['deployment_decision']; m = report['monitoring']; a = report['alerts']; r = report['response']
    name = report['intervention']['name']; method = LABELS[name]; rep = report['reproducibility']
    policy = report['settings']['policy']; max_edits = report['threat_model']['attack_budget']
    sel = report['intervention']['validation_selection']; sim = m['simulated_regression']
    readings = table_readings(report)
    typos = 'no character edits' if max_edits == 0 else f"up to {max_edits} character edit{'s' if max_edits > 1 else ''} per query"
    table = lambda frame, reading: ['| ' + ' | '.join(frame.columns) + ' |', '|' + '---|'*len(frame.columns),
                           *['| ' + ' | '.join(map(str, row)) + ' |' for row in frame.itertuples(index=False)],
                           '', reading]
    rows = ['# Retrieval robustness to query typos: evaluation and monitoring', '', report['tutorial'], '', report['scope'], '',
            '## Summary', '', report_summary(report), '',
            '## Experiment settings', '', '| Setting | Value |', '|---|---|',
            f'| Candidate | {method} |',
            f"| Queries | {s['n_evaluated']} BEIR SciFact test queries with {typos} |",
            f"| Search | Dense search over {report['dataset']['corpus_count']:,} documents; 20 candidates; top 10 scored; reranker off |",
            f"| Repair adapter | fit on {len(sel['fit_query_ids'])} training queries; λ = {sel['lambda']:g} chosen on "
            f"{len(sel['selection_query_ids'])} validation queries |",
            f"| Gate policy | clean drop tolerance = {policy['clean_drop_tolerance']:g} |",
            f"| Quality alert | average score loss > {policy['quality_tolerance']:g}, or any severe query loss |",
            '', readings['settings'], '', report['settings']['policy_note'], '',
            '## Deployment decision', '', f"Decision: {d['status']}. {d['reason']}", '',
            f"Release rule: {d['gate']} Severe original-query loss: {d['policy']['broken_query_rule']}.", '',
            *table(decision_table(report), readings['decision']), '',
            '## Clean and typo performance', '', 'Average scores on the same queries; higher nDCG@10 and Recall@10 are better.', '',
            *table(performance_table(report, candidate_only=True), readings['evaluation']), '',
            f'## Confidence intervals: {method}', '', '| Measure | Estimate [95% CI] |', '|---|---:|',
            f"| ΔnDCG@10 after typos, within {method} (after − before) | {_ci(report['evaluation']['paired_ci'])} |",
            f"| Gain over original search, after typos | {_ci(c['shifted_difference'])} |",
            f"| Gain over original search, original input | {_ci(c['clean_difference'])} |"]
    if name != 'none':
        g = report['evaluation']['intervention_gains'][name]
        rows += [f"| Recall@10 gain over original search, after typos | {_ci(g['typo_recall_gain'])} |",
                 f"| Recall@10 gain over original search, original input | {_ci(g['clean_recall_change'])} |"]
    rows += ['', readings['confidence'], '',
             'The 95% intervals use 500 bootstrap samples of query IDs, keeping each query\'s two scores together. '
             'An interval entirely above zero supports a gain. An interval including zero does not establish a gain.', '',
             '## Monitoring', '', f"{m['scope']} System in use: {m['monitored_label']}. "
             f"Fixed test set: {m['canary_set']['query_count']} {m['canary_set']['description']}. "
             f"Reference: {m['reference_definition']}.", '',
             *table(pd.DataFrame(_canary_rows(report, ['Saved reference', 'Current run',
                                                       'Simulated regression', 'After rollback'])), readings['verification']), '',
             '## Quality alert', '', *table(alert_table(report), readings['alerts']), '',
             f"After rollback: {a['after_rollback']['status'].lower()}. Simulated regression: "
             f"{sim['story']} It affected "
             f"{len(sim['affected_query_ids'])} of {sim['misspelled_count']} typo queries in a copy of the quality-check results.", '',
             '## Representative failure cases', '']
    for repair in REPAIRS:
        cases = report['evaluation']['intervention_gains'][repair]['broken_original_queries']
        if cases:
            case = cases[0]; words = ', '.join(f'{x} → {y}' for x, y in case['changed_words'])
            rows.append(f"- Severe original-query loss, {LABELS[repair]}: nDCG@10 "
                        f"{case['reference_ndcg']:.4f} with original search → {case['current_ndcg']:.4f}. "
                        f"Query: {case['query_text']}" + (f' Correction: {words}.' if words else ''))
    for case in report['failure_examples']:
        rows.append(f"- Typo damage, {method}: nDCG@10 {case['clean_ndcg']:.4f} → "
                    f"{case['shifted_ndcg']:.4f}. Original: {case['original_query']} After typos: {case['shifted_query']}")
    for case in sim['severe_failures'][:2]:
        rows.append(f'- Simulated regression: {_example_text(case)}')
    rows += ['', '## Response plan', '', f"Owner: {r['owner']}.", '', *table(response_plan_table(report), readings['response']), '',
             f"Simulated response. Diagnose: {r['diagnosis']['finding']} Mitigate: {r['mitigation']['action']} "
             f"{r['mitigation']['gate']} Verify: rerun {r['verification']['rerun']}; "
             f"{len(r['verification']['recovered_query_ids'])} failed queries recovered, "
             f"{r['verification']['remaining_severe_failures']} severe failures remain, and the status is "
             f"{r['verification']['status'].lower()}.",
             '', '## Next steps', '', *[f'- {step}' for step in report['next_steps']], '',
             '## Reproduction', '',
             'robustness_report.json holds every per-query score, the train/validation/test query IDs, '
             'model versions, data checksums, and the quality-check measurements needed to reproduce this run.', '',
             'Models: ' + '; '.join(rep['model_revisions']) +
             f". Dataset: {report['dataset']['name']}, {report['dataset']['corpus_count']:,} documents. "
             'Adjusting only a tolerance raises or lowers a decision cutoff; the measured scores and rankings stay fixed.',
             '', '## Appendix: Ranking stability', '',
             'Optional original/typo ranking agreement; lower drift means more stable lists, not necessarily better evidence.', '',
             *table(ranking_stability_table(report), readings['ranking'])]
    from .presentation import plain_markdown
    return plain_markdown('\n'.join(rows))


def save_report(report, directory=None):
    folder = ROOT/'outputs' if directory is None else Path(directory)
    if not folder.resolve().is_relative_to((ROOT/'outputs').resolve()):
        raise ValidationError('Participant reports may only be written under resources/outputs/.')
    md = report_markdown(report); js = json.dumps(report, indent=2, allow_nan=False)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'robustness_report.md').write_text(md, encoding='utf-8')
        (folder/'robustness_report.json').write_text(js, encoding='utf-8')
        return {'saved': True, 'markdown': md, 'json': js, 'message': 'Saved robustness_report.md and .json under resources/outputs/.'}
    except OSError as error:
        return {'saved': False, 'markdown': md, 'json': js,
                'message': f'Could not save both report files: {error}. Use the displayed Markdown and download links.'}


def _section(anchor, title, frame, note='', interpretation=None):
    from html import escape
    return (f'<section id="{anchor}"><h4>{escape(title)}</h4>' + (f'<p>{escape(note)}</p>' if note else '') +
            frame.to_html(index=False, border=0, escape=True) +
            (interpretation_html(interpretation) if interpretation is not None else '') + '</section>')


def show_evaluation(report):
    """Evaluate only the candidate and original search; M4 compares all repair means."""
    from IPython.display import display, HTML
    from . import plotting as viz
    viz.show(viz.report_figure(report))
    s = report['evaluation']['mean_and_tail']
    display(HTML(_section('m6-evaluation', 'Search quality before and after typos',
                          performance_table(report, candidate_only=True), f"{s['n_evaluated']} test queries; maximum edits: "
                          f"{report['threat_model']['attack_budget']}; higher nDCG@10 and Recall@10 are better",
                          interpretation=table_readings(report)['evaluation'])))
    show_interpretation(takeaways(report)['evaluation'])


def show_decision(report):
    """Deployment decision: the gate for every repair, then the candidate's Go/Hold and reason."""
    from IPython.display import display, HTML, Markdown
    d = report['deployment_decision']
    display(HTML(_section('m6-decision', f"Deployment decision for {d['candidate_label']}: {d['status']}",
                          decision_table(report),
                          'Higher gains are better. No original query may suffer a severe loss. Go requires all three checks to pass.',
                          interpretation=table_readings(report)['decision'])))
    lines = [f"Decision: {d['status']}. {d['reason']}"]
    lines += [f"Decision: {other['status']}. {other['reason']}" for method, other in d['by_method'].items()
              if method != d['candidate'] and other['checks'][0]['passed']]
    display(Markdown('\n\n'.join(lines)))
    show_interpretation(takeaways(report)['decision'])


def show_monitoring(report):
    """Repeat the quality check against saved scores, leaving ranking diagnostics to the appendix."""
    from IPython.display import display, HTML, Markdown
    m = report['monitoring']; d = report['deployment_decision']; observed = m['observed']
    canary = m['canary_set']
    display(HTML(_section('m6-monitoring', f"Quality check: {m['monitored_label']}",
                          monitoring_table(report), f"{canary['query_count']} test queries, original and with "
                          f"their fixed typos (maximum edits: {canary['severity']}); higher quality scores and fewer severe failures are better",
                          interpretation=table_readings(report)['monitoring'])))
    why = ('the system kept in use' if d['approved_system'] == 'none' else 'selected after a Go decision')
    lines = [f"Monitored system: {m['monitored_label'].lower()}, {why}." +
             (f" {d['candidate_label']} is on hold, so it is not deployed or monitored."
              if d['candidate'] != 'none' and d['status'] == 'Hold' else '')]
    display(Markdown('\n\n'.join(lines)))
    show_interpretation(takeaways(report)['monitoring'])


def show_alerts(report):
    """Show one quality alert combining average score loss and severe query losses."""
    from IPython.display import display, HTML, Markdown
    a = report['alerts']; sim = report['monitoring']['simulated_regression']
    display(HTML(_section('m6-alerts', 'One quality alert, two checks', alert_table(report),
                          f"quality tolerance = {a['policy']['quality_tolerance']:g}; "
                          'negative average changes are losses, and fewer severe failures are better',
                          interpretation=table_readings(report)['alerts'])))
    if not sim['misspelled_count']:
        display(Markdown('Severity 0 adds no typos, so the faulty step never runs.'))
    show_interpretation(takeaways(report)['alerts'])


def show_response(report):
    """Inspect one failure, roll back the faulty update, and rerun the quality check."""
    from IPython.display import display, HTML, Markdown
    r = report['response']; sim = report['monitoring']['simulated_regression']; a = report['alerts']
    if not a['simulated_regression']['alert']:
        display(Markdown('No alert fired for the simulated regression, so there is nothing to diagnose. ' +
                         ('Severity 0 adds no typos, so the faulty typo-handling step never runs.'
                          if not sim['misspelled_count'] else '')))
    else:
        changed, change = sim['changed_rankings'], sim['quality_change']
        lines = [f"Changed results: {changed['typo']} typo queries and {changed['original']} original queries.",
                 f"Mean typo-score changes: nDCG@10 {change['typo_ndcg']:+.4f}; "
                 f"Recall@10 {change['typo_recall']:+.4f}. Severe failures: {len(sim['severe_failures'])}.",
                 f"Check: {r['diagnosis']['finding']} Owner: {r['diagnosis']['owner']}."]
        if sim['severe_failures']:
            lines.append('Example: ' + _example_text(sim['severe_failures'][0]))
        lines += [f"Roll back: {r['mitigation']['action']} {r['mitigation']['gate']}",
                  f"Verify: rerun {r['verification']['rerun']}."]
        display(Markdown('\n\n'.join(lines)))
        display(HTML(_section('m6-verify', 'Recheck quality after rollback', verification_table(report),
                              'Higher quality scores and fewer severe failures are better.',
                              interpretation=table_readings(report)['verification'])))
        show_interpretation(takeaways(report)['verification'])


def show_ranking_reference(report):
    """Display ranking agreement only in M6's optional appendix."""
    from IPython.display import display, HTML
    display(HTML(_section('m6-ranking-diagnostics', 'Optional ranking stability', ranking_stability_table(report),
                          'Original versus typo top 10 within each run; lower drift means more agreement, not better evidence',
                          interpretation=table_readings(report)['ranking'])))
    show_interpretation(takeaways(report)['ranking'])


def show_downloads(saved):
    """Report section: download links for the saved Markdown and JSON reports."""
    import base64
    from html import escape
    from IPython.display import display, HTML
    links = []
    for ext, mime, text in [('md', 'text/markdown', saved['markdown']), ('json', 'application/json', saved['json'])]:
        payload = base64.b64encode(text.encode('utf-8')).decode('ascii')
        links.append(f'<a download="robustness_report.{ext}" href="data:{mime};base64,{payload}">Download .{ext}</a>')
    display(HTML('<p>'+' &nbsp; | &nbsp; '.join(links)+'</p><p>'+escape(saved['message'])+'</p>'))


def show_report_reference(report):
    """Display complete audit details in the notebook's optional reference cell."""
    from IPython.display import display, Markdown
    display(Markdown(report_markdown(report)))
