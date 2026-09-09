# -*- coding: utf-8 -*-
"""Prompt ensembling: score every cell under several paraphrased system prompts.

WHY THIS EXISTS
---------------
HANDOVER §10.3 lists this as the most promising cheap idea left, and the reasoning
is sound: test-time symmetry averaging -- scoring both presentation orders and
averaging -- was the biggest late gain in the project (F1 0.523 -> 0.554, AUC
0.792 -> 0.817) for seven minutes of GPU and nothing to tune. It works because two
views of the same cell make partly independent errors. A paraphrased system prompt
is another such view.

The paraphrases restate the CRITERION.md v2 rule without altering it: N only when a
clause makes the pairing impossible or means it never occurs, and ineffectiveness is
never grounds for N. If a variant changed the criterion it would not be measuring
the same task.

Each variant is scored over BOTH presentation orders (dataset_val_cellsym.jsonl
carries them under a shared `pair` label), so a variant's own score is already the
deployed TTA score. The ensemble then averages across variants on top of that.

Every variant is dumped separately, so combinations can be re-evaluated later with
no GPU (by_code.py and ensemble.py both read these files).

Usage:
    python prompt_ensemble.py --adapter results/cell_w8sym --batch 16
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from calibrate import PREFIX, metrics, sweep, verdict_token_ids, load_model  # noqa: E402
from by_code import auc  # noqa: E402

# v0 is the prompt the model was fine-tuned on. v1 and v2 restate the same rule.
VARIANTS = [
    "You are a defence and security analyst carrying out cross-consistency assessment "
    "for General Morphological Analysis. You are given two options from different facets "
    "of ONE scenario, with what each requires and provides. Decide whether a single real "
    "scenario could contain both. Answer N only when you can name the clause that makes "
    "it impossible or means it never occurs; ineffective or unwise is still Y.",

    "You are an analyst performing cross-consistency assessment within General "
    "Morphological Analysis. Below are two values drawn from different parameters of the "
    "SAME scenario, each listed with its requirements and the properties it supplies. "
    "Judge whether any one internally consistent scenario could contain both at once. "
    "Answer N only if a specific clause makes the pairing impossible, or means it never "
    "arises in practice; a combination that merely works badly is still Y.",

    "Cross-consistency assessment. Two options are given, taken from separate facets of a "
    "SINGLE scenario, each with what it REQUIRES and what it PROVIDES. The question is "
    "whether any real configuration exists that contains both. Answer N only when a named "
    "clause rules the pairing out, or means no real case instantiates it. Ineffectiveness, "
    "inefficiency and poor design are not grounds for N.",
]


def score_variant(tok, model, rows, system, batch):
    """P(N) for every row, with the system message replaced by `system`."""
    y_id, n_id = verdict_token_ids(tok)
    out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        prompts = []
        for r in chunk:
            msgs = [dict(m) for m in r['messages'][:-1]]
            if system is not None and msgs and msgs[0]['role'] == 'system':
                msgs[0]['content'] = system
            prompts.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True) + PREFIX)
        enc = tok(prompts, return_tensors='pt', padding=True).to(model.device)
        with torch.no_grad():
            # logits_to_keep=1 runs lm_head over the final position only. calibrate.py
            # materialises [batch, seq, 151936] and then discards all but the last row,
            # which at batch 16 is ~1.2 GB and puts the card at 96% occupancy -- the
            # condition §7 of HANDOVER.md blames for two bugchecks. Same numbers, and
            # left padding still guarantees the last position is the real final token.
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
        pair = torch.stack([logits[:, y_id], logits[:, n_id]], dim=-1).float()
        out += torch.softmax(pair, dim=-1)[:, 1].tolist()
        if (i // batch) % 25 == 0:
            print('      %d/%d' % (min(i + batch, len(rows)), len(rows)), flush=True)
    return out


def tta_average(rows, scores):
    """Collapse the two presentation orders into one score per (domain, pair)."""
    acc = {}
    for r, s in zip(rows, scores):
        acc.setdefault((r['domain'], r['pair']), []).append(s)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def report(name, per_cell, truth, dev_doms):
    keys = sorted(per_cell)
    dev = [k for k in keys if k[0] in dev_doms]
    test = [k for k in keys if k[0] not in dev_doms]
    a_all = auc([per_cell[k] for k in keys], [truth[k] for k in keys])
    a_test = auc([per_cell[k] for k in test], [truth[k] for k in test])
    thr = sweep([per_cell[k] for k in dev], [truth[k] for k in dev])[0]
    m = metrics(['N' if per_cell[k] >= thr else 'Y' for k in test],
                [truth[k] for k in test])
    print('  %-22s AUC(all) %.3f   AUC(test) %.3f   thr %.3f   '
          'test F1 %.3f  bal %.3f' % (name, a_all, a_test, thr, m['f1'],
                                      m['balanced_accuracy']))
    return a_all, a_test, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', default=os.path.join(HERE, 'results', 'cell_w8sym'))
    ap.add_argument('--data', default=os.path.join(HERE, 'dataset_val_cellsym.jsonl'))
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--prefix', default='scores_pe')
    ap.add_argument('--keep-system', action='store_true',
                    help="score each row with its OWN system message instead of "
                         "substituting a variant. Required for datasets whose prompt "
                         "differs structurally, e.g. retrieval-augmented rows that "
                         "carry precedents and a matching system message.")
    ap.add_argument('--variants', type=int, default=0,
                    help='use only the first N system prompts. --variants 1 makes '
                         'this a plain TTA scorer (v0 is the prompt the model was '
                         'trained on), which is how a new adapter is scored on the '
                         'deployed configuration.')
    a = ap.parse_args()

    variants = [None] if a.keep_system else (
        VARIANTS[:a.variants] if a.variants else VARIANTS)
    rows = [json.loads(l) for l in open(a.data, encoding='utf-8')]
    truth = {(r['domain'], r['pair']): r['label'] for r in rows}
    doms = sorted({r['domain'] for r in rows})
    dev_doms = set(doms[::2])            # same split rule as calibrate.py
    print('  %d rows (%d cells x both orders), %d domains, %d prompt variants'
          % (len(rows), len(truth), len(doms), len(variants)))

    print('  loading %s' % a.adapter)
    tok, model = load_model(a.adapter)

    per_variant = []
    for i, sysmsg in enumerate(variants):
        print('\n  variant v%d ...' % i, flush=True)
        s = score_variant(tok, model, rows, sysmsg, a.batch)
        cells = tta_average(rows, s)
        per_variant.append(cells)
        path = os.path.join(HERE, '%s_v%d.json' % (a.prefix, i))
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([{'domain': d, 'pair': p, 'label': truth[(d, p)],
                        'p_inconsistent': v,
                        'split': 'dev' if d in dev_doms else 'test'}
                       for (d, p), v in cells.items()], f, indent=1)
        print('    wrote %s' % os.path.basename(path))

    print('\n' + '=' * 78)
    print('  PROMPT ENSEMBLE  (each variant already averaged over both orders)')
    print('=' * 78)
    for i, cells in enumerate(per_variant):
        report('v%d alone' % i, cells, truth, dev_doms)

    keys = per_variant[0].keys()
    if len(per_variant) == 1:
        return
    mean_all = {k: sum(v[k] for v in per_variant) / len(per_variant) for k in keys}
    print()
    report('ensemble of %d' % len(per_variant), mean_all, truth, dev_doms)
    # v0 is the deployed configuration, so this pairing is the honest comparison
    pair01 = {k: (per_variant[0][k] + per_variant[1][k]) / 2 for k in keys}
    pair02 = {k: (per_variant[0][k] + per_variant[2][k]) / 2 for k in keys}
    report('v0+v1', pair01, truth, dev_doms)
    report('v0+v2', pair02, truth, dev_doms)


if __name__ == '__main__':
    main()
