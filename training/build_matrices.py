# -*- coding: utf-8 -*-
"""Expands authored source files into validated matrices.

The source format lists ONLY the exclusions, each with a typed code and verbatim
citations. Every unlisted cell is Y by construction. This is deliberate: under
CRITERION.md v2 an N is a claim that requires evidence, so the format makes an
unjustified N impossible to express rather than merely discouraged.

  python build_matrices.py                 # build + validate
  python build_matrices.py --check         # validate only, no write
"""
import argparse, glob, itertools, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'authoring', 'src')
OUT = os.path.join(HERE, 'matrices_v2')
CODES = {'N1': 'definitional', 'N2': 'physical', 'N3': 'capability',
         'N4': 'temporal', 'N5': 'authority', 'N6': 'scope', 'N7': 'implausible'}


def norm(s):
    return ' '.join((s or '').lower().split())


def build_one(m, path, errs):
    dom = m.get('domain', '?')
    params = m.get('parameters', {})
    chars = m.get('characterisations', {})

    def err(msg):
        errs.append('%s [%s]: %s' % (os.path.basename(path), dom, msg))

    # A migrated matrix whose candidates have not yet been re-decided would emit an
    # all-Y grid, which is wrong data rather than missing data. Refuse to build it.
    if m.get('candidates'):
        err('still has %d v1 candidates awaiting re-decision under CRITERION.md v2'
            % sum(len(v) for v in m['candidates'].values()))

    if len(params) < 3:
        err('needs at least 3 parameters, has %d' % len(params))
    for p, vs in params.items():
        if len(vs) < 3:
            err('parameter "%s" has only %d values' % (p, len(vs)))
        if len(set(vs)) != len(vs):
            err('parameter "%s" has duplicate values' % p)
        for v in vs:
            c = chars.get(v)
            if not c:
                err('value "%s" has no characterisation' % v); continue
            for f in ('requires', 'provides'):
                if not (c.get(f) or '').strip():
                    err('value "%s" has empty %s' % (v, f))

    names = [v for vs in params.values() for v in vs]
    if len(set(names)) != len(names):
        err('the same value name appears under two parameters')
    for v in chars:
        if v not in names:
            err('characterisation "%s" belongs to no parameter' % v)

    verdicts, reasons = {}, {}
    keys = ['%s|%s' % (a, b) for a, b in itertools.combinations(list(params), 2)]
    for key in keys:
        pa, pb = key.split('|')
        ra, cb = params[pa], params[pb]
        grid = [['Y'] * len(cb) for _ in ra]
        rk = {}
        for e in m.get('exclusions', {}).get(key, []):
            r, c = e.get('row'), e.get('col')
            if r not in ra:
                err('"%s": exclusion row "%s" is not a value of %s' % (key, r, pa)); continue
            if c not in cb:
                err('"%s": exclusion col "%s" is not a value of %s' % (key, c, pb)); continue
            i, j = ra.index(r), cb.index(c)
            if grid[i][j] == 'N':
                err('"%s": duplicate exclusion for %s x %s' % (key, r, c)); continue
            code = e.get('code')
            if code not in CODES:
                err('"%s": %s x %s has invalid code "%s"' % (key, r, c, code)); continue
            hay = norm(' '.join([chars.get(r, {}).get('requires', ''),
                                 chars.get(r, {}).get('provides', ''),
                                 chars.get(c, {}).get('requires', ''),
                                 chars.get(c, {}).get('provides', '')]))
            bad = False
            for f in ('requires', 'blocked_by'):
                q = norm(e.get(f, ''))
                if not q:
                    err('"%s": %s x %s missing %s' % (key, r, c, f)); bad = True
                elif q not in hay:
                    err('"%s": %s x %s cites "%s" which is not in either characterisation'
                        % (key, r, c, (e.get(f) or '')[:44])); bad = True
            if bad:
                continue
            grid[i][j] = 'N'
            rk['%d,%d' % (i + 1, j + 1)] = {'code': code, 'requires': e['requires'],
                                            'blocked_by': e['blocked_by']}
        verdicts[key] = [''.join(r) for r in grid]
        if rk:
            reasons[key] = rk

    for key in m.get('exclusions', {}):
        if key not in verdicts:
            err('exclusions given for unknown parameter pair "%s"' % key)

    return {'domain': dom, 'group': m.get('group', ''), 'parameters': params,
            'characterisations': chars, 'verdicts': verdicts,
            'verdict_reasons': reasons, 'notes': m.get('notes', {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(SRC, '*.json')))
    if not files:
        print('  no source files in %s' % SRC); return 1

    errs, out, doms = [], [], set()
    for f in files:
        try:
            data = json.load(open(f, encoding='utf-8'))
        except Exception as e:
            errs.append('%s: unreadable (%s)' % (os.path.basename(f), e)); continue
        for m in (data if isinstance(data, list) else [data]):
            d = m.get('domain', '?')
            if d in doms:
                errs.append('%s: duplicate domain "%s"' % (os.path.basename(f), d))
            doms.add(d)
            out.append(build_one(m, f, errs))

    cells = sum(len(r) for m in out for g in m['verdicts'].values() for r in g)
    n = sum(r.count('N') for m in out for g in m['verdicts'].values() for r in g)
    just = sum(len(v) for m in out for v in m['verdict_reasons'].values())

    print('=' * 68)
    print('  BUILD  (CRITERION.md v2)')
    print('=' * 68)
    print('  source files      %d' % len(files))
    print('  matrices          %d' % len(out))
    print('  parameter pairs   %d' % sum(len(m['verdicts']) for m in out))
    print('  cells             %d' % cells)
    print('  N cells           %d  (%.1f%%)' % (n, 100.0 * n / max(cells, 1)))
    print('  N justified       %d  (%.1f%%)' % (just, 100.0 * just / max(n, 1)))
    print('  errors            %d' % len(errs))
    for e in errs[:15]:
        print('    - %s' % e)
    if errs:
        print('\n  FAIL'); return 1
    if not a.check:
        os.makedirs(OUT, exist_ok=True)
        for old in glob.glob(os.path.join(OUT, '*.json')):
            os.remove(old)
        json.dump(out, open(os.path.join(OUT, 'corpus.json'), 'w', encoding='utf-8'),
                  indent=1, ensure_ascii=False)
        print('\n  wrote %s' % os.path.join(OUT, 'corpus.json'))
    print('  PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
