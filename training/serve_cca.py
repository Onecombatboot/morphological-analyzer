# -*- coding: utf-8 -*-
"""Local HTTP service exposing the fine-tuned cross-consistency model.

WHY A SERVICE RATHER THAN A GGUF
--------------------------------
The Spring app runs a GGUF through llama.cpp (gma.llm.model). What this project
produced is a PEFT LoRA adapter on Qwen2.5-3B in HuggingFace format, and there is
no GGUF writer on this machine (the `gguf` package is absent, which is also why
gguf_to_hf.py had to be written by hand). Converting would mean merging the
adapter, writing a GGUF encoder and a Q4_K quantiser from scratch. Serving the
model where it already runs is the same result for a fraction of the work, and it
keeps the Python evaluation path and the deployed path byte-identical -- the
prompt here is imported from build_dataset, not retyped.

THE INPUT THE MODEL NEEDS
The Spring endpoint /api/matrix/full-analysis posts {parameter: [values]} and
nothing else. This model cannot work from names alone: it needs each value's
REQUIRES and PROVIDES. That is not a limitation to work around, it is the finding
that made the project work -- the 'grid' format, which asked the model to invent
its own premises, scored at chance (HANDOVER section 4.4). So /cca requires
characterisations, and /score exists for callers that already have them.

ENDPOINTS
    GET  /health              model, adapter, threshold, device
    POST /score               {"pairs":[{"a","a_requires","a_provides",
                                         "b","b_requires","b_provides",
                                         "pa","pb"}, ...]}
                              -> {"scores":[p_inconsistent, ...]}
    POST /cca                 {"domain","parameters","characterisations"}
                              -> full verdict grids, solution space, evidence

Every score averages BOTH presentation orders, which is what the deployed
configuration does (F1 0.523 -> 0.554 when it was added).

Usage:
    python serve_cca.py --adapter results/cell_r32d --port 8000
"""
import argparse
import itertools
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STATE = {}
LOCK = threading.Lock()          # one CUDA context, so serialise requests


def load(adapter, base_only=False):
    import torch
    import calibrate
    from build_dataset import SYSTEM_CELL
    tok, model = calibrate.load_model(None if base_only else adapter)
    y_id, n_id = calibrate.verdict_token_ids(tok)
    STATE.update(tok=tok, model=model, y_id=y_id, n_id=n_id,
                 system=SYSTEM_CELL, prefix=calibrate.PREFIX, torch=torch,
                 adapter=adapter)
    print('  loaded %s  (verdict ids Y=%d N=%d)' % (adapter, y_id, n_id), flush=True)


def _one(a, ca, b, cb, pa, pb):
    """P(inconsistent) for one presentation order."""
    torch = STATE['torch']
    tok, model = STATE['tok'], STATE['model']
    # identical to build_dataset.cell_examples(); the model was trained on this
    # exact wording and any drift silently degrades it
    user = '\n'.join([
        'Facet A — %s' % pa,
        '%s | REQUIRES: %s | PROVIDES: %s' % (a, ca['requires'], ca['provides']),
        '',
        'Facet B — %s' % pb,
        '%s | REQUIRES: %s | PROVIDES: %s' % (b, cb['requires'], cb['provides']),
        '',
        'Can one scenario contain both?'])
    msgs = [{'role': 'system', 'content': STATE['system']},
            {'role': 'user', 'content': user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True) + STATE['prefix']
    enc = tok(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        lg = model(**enc, logits_to_keep=1).logits[0, -1, :]
    pair = torch.stack([lg[STATE['y_id']], lg[STATE['n_id']]]).float()
    return torch.softmax(pair, dim=-1)[1].item()


def score_pair(a, ca, b, cb, pa, pb):
    """Both orders averaged -- cross-consistency is symmetric."""
    return 0.5 * (_one(a, ca, b, cb, pa, pb) + _one(b, cb, a, ca, pb, pa))


CHAR_SYSTEM = (
    "You describe options in a scenario-planning matrix. For the option you are given, "
    "state two things in plain noun phrases, comma separated, no sentences:\n"
    "REQUIRES: what a scenario must already provide for this option to be present\n"
    "PROVIDES: what this option contributes or entails, including what it rules out\n"
    "Be concrete and specific to the option. Three to six short clauses each."
)


def characterise(param, value, max_new=90):
    """Generate a requires/provides characterisation for one value.

    Generated with the adapter DISABLED, i.e. from the base model. The adapter was
    trained to emit verdicts and would otherwise pull the generation toward
    'VERDICT: ...' rather than a description. One model in memory, two behaviours.
    """
    torch = STATE['torch']
    tok, model = STATE['tok'], STATE['model']
    user = ('Scenario facet: %s\nOption: %s\n\nGive REQUIRES and PROVIDES for this option.'
            % (param, value))
    msgs = [{'role': 'system', 'content': CHAR_SYSTEM},
            {'role': 'user', 'content': user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors='pt').to(model.device)
    ctx = model.disable_adapter() if hasattr(model, 'disable_adapter') else None
    try:
        if ctx is not None:
            ctx.__enter__()
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    txt = tok.decode(g[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)

    req = prov = None
    for line in txt.splitlines():
        line = line.strip().lstrip('-* ')
        up = line.upper()
        if up.startswith('REQUIRES'):
            req = line.split(':', 1)[-1].strip()
        elif up.startswith('PROVIDES'):
            prov = line.split(':', 1)[-1].strip()
    # never return an empty characterisation: the scorer would then be judging
    # from nothing, which is the failure mode the whole cell format exists to avoid
    if not req:
        req = 'the conditions under which %s applies' % value.lower()
    if not prov:
        prov = 'the properties %s brings to the scenario' % value.lower()
    return {'requires': req, 'provides': prov}


def assess_box(parameters, model_thr, chars=None):
    """Characterise every value if needed, then score every cross-parameter pair."""
    chars = dict(chars or {})
    generated = []
    for pname, vals in parameters.items():
        for v in vals:
            if v not in chars or not chars[v].get('requires'):
                chars[v] = characterise(pname, v)
                generated.append(v)

    names = list(parameters)
    pairs = []
    for pa, pb in itertools.combinations(names, 2):
        for a in parameters[pa]:
            for b in parameters[pb]:
                p = score_pair(a, chars[a], b, chars[b], pa, pb)
                pairs.append({'pa': pa, 'a': a, 'pb': pb, 'b': b,
                              'p_inconsistent': round(p, 4),
                              'contradiction': bool(p >= model_thr)})
    return {'characterisations': chars, 'generated': generated,
            'pairs': pairs, 'threshold': model_thr}


def run_cca(box, model_thr, rule_threshold=0.10, wide=True):
    import cca
    ctx = {'score_one': score_pair}
    verdicts = cca.assess(box, rule_threshold, wide, ctx, model_thr)
    raw, kept = cca.solution_space(box, verdicts)
    out = {}
    for key, grid in verdicts.items():
        out[key] = [[{'a': c['a'], 'b': c['b'], 'verdict': c['verdict'],
                      'source': c['source'], 'evidence': c['evidence']}
                     for c in row] for row in grid]
    n_n = sum(1 for g in verdicts.values() for r in g for c in r if c['verdict'] == 'N')
    n_c = sum(len(r) for g in verdicts.values() for r in g)
    return {'domain': box.get('domain', ''), 'grids': out,
            'raw_configurations': raw, 'consistent_configurations': kept,
            'eliminated_pct': round(100.0 * (raw - kept) / raw, 2) if raw else 0.0,
            'cells': n_c, 'excluded_cells': n_n,
            'model_threshold': model_thr}


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def _send(self, code, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(200, {})

    def log_message(self, fmt, *args):
        print('  %s - %s' % (self.address_string(), fmt % args), flush=True)

    def do_GET(self):
        if self.path.rstrip('/') in ('/health', ''):
            self._send(200, {'status': 'ok', 'adapter': STATE.get('adapter'),
                             'base': 'Qwen2.5-3B + LoRA',
                             'default_threshold': STATE.get('threshold'),
                             'device': str(STATE['model'].device)})
        else:
            self._send(404, {'error': 'no such endpoint'})

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        try:
            req = json.loads(self.rfile.read(n) or b'{}')
        except Exception as e:
            return self._send(400, {'error': 'bad JSON: %s' % e})
        path = self.path.rstrip('/')
        t0 = time.time()
        try:
            with LOCK:
                if path == '/score':
                    pairs = req.get('pairs') or []
                    scores = [score_pair(p['a'], {'requires': p['a_requires'],
                                                  'provides': p['a_provides']},
                                         p['b'], {'requires': p['b_requires'],
                                                  'provides': p['b_provides']},
                                         p.get('pa', 'Facet A'), p.get('pb', 'Facet B'))
                              for p in pairs]
                    return self._send(200, {'scores': [round(s, 4) for s in scores],
                                            'seconds': round(time.time() - t0, 2)})
                if path == '/characterise':
                    params = req.get('parameters') or {}
                    out = {}
                    for pname, vals in params.items():
                        for v in vals:
                            out[v] = characterise(pname, v)
                    return self._send(200, {'characterisations': out,
                                            'seconds': round(time.time() - t0, 2)})
                if path == '/assess-box':
                    params = req.get('parameters') or {}
                    if not params:
                        return self._send(400, {'error': "missing 'parameters'"})
                    thr = float(req.get('threshold', STATE['threshold']))
                    res = assess_box(params, thr, req.get('characterisations'))
                    res['seconds'] = round(time.time() - t0, 2)
                    return self._send(200, res)
                if path == '/cca':
                    for k in ('parameters', 'characterisations'):
                        if k not in req:
                            return self._send(400, {
                                'error': "missing '%s'" % k,
                                'hint': 'this model scores from each value\'s REQUIRES '
                                        'and PROVIDES; names alone are not enough'})
                    missing = [v for vals in req['parameters'].values() for v in vals
                               if v not in req['characterisations']]
                    if missing:
                        return self._send(400, {'error': 'values without a '
                                                'characterisation', 'values': missing[:20]})
                    thr = float(req.get('threshold', STATE['threshold']))
                    res = run_cca(req, thr)
                    res['seconds'] = round(time.time() - t0, 2)
                    return self._send(200, res)
            return self._send(404, {'error': 'no such endpoint'})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._send(500, {'error': str(e)})


def main():
    ap = argparse.ArgumentParser()
    # cell_r32d is the deployed model: rank-32 QLoRA on the clause-grounded criterion,
    # the one every published figure refers to (F1 0.828 at the deployed operating
    # point, AUC 0.878). Changing this default silently changes the served model.
    ap.add_argument('--adapter', default=os.path.join(HERE, 'results', 'cell_r32d'))
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--threshold', type=float, default=0.485,
                    help='P(inconsistent) at or above which a cell is marked N')
    a = ap.parse_args()
    STATE['threshold'] = a.threshold
    print('  loading model...', flush=True)
    load(a.adapter)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print('  listening on http://%s:%d  (GET /health, POST /score, POST /cca)'
          % (a.host, a.port), flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
