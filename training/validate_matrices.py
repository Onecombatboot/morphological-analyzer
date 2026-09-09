# -*- coding: utf-8 -*-
"""Hard validation of the matrix corpus against CRITERION.md (v2).

The point is that a label cannot drift away from its evidence. Every N must name a
typed reason and quote the clause it rests on, and that quotation is re-checked
against the characterisation text on every build. A label nobody can justify is a
build failure, not a warning.

  python validate_matrices.py            # report
  python validate_matrices.py --strict   # exit 1 if anything fails
"""
import argparse, glob, json, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
MAT = os.path.join(HERE, 'matrices')
CODES = {'N1': 'definitional', 'N2': 'physical', 'N3': 'capability',
         'N4': 'temporal', 'N5': 'authority', 'N6': 'scope', 'N7': 'implausible'}


def norm(s):
    return ' '.join((s or '').lower().split())


def check(m, mi):
    errs, warns, missing = [], [], 0
    tag = 'matrix %d (%s)' % (mi, m.get('domain', '?'))

    params = m.get('parameters', {})
    chars = m.get('characterisations', {})
    reasons = m.get('verdict_reasons', {})

    for p, vs in params.items():
        for v in vs:
            if v not in chars:
                errs.append('%s: value "%s" has no characterisation' % (tag, v))
            else:
                for f in ('requires', 'provides'):
                    if not chars[v].get(f):
                        errs.append('%s: "%s" has empty %s' % (tag, v, f))

    for key, grid in m.get('verdicts', {}).items():
        if '|' not in key:
            errs.append('%s: bad verdict key "%s"' % (tag, key)); continue
        pa, pb = key.split('|')
        if pa not in params or pb not in params:
            errs.append('%s: verdict key "%s" names an unknown parameter' % (tag, key)); continue
        ra, cb = params[pa], params[pb]
        if len(grid) != len(ra):
            errs.append('%s: "%s" has %d rows, expected %d' % (tag, key, len(grid), len(ra)))
            continue
        rk = reasons.get(key, {})
        for i, row in enumerate(grid):
            if len(row) != len(cb):
                errs.append('%s: "%s" row %d has %d cells, expected %d'
                            % (tag, key, i + 1, len(row), len(cb)))
                continue
            for j, c in enumerate(row):
                if c not in 'YN':
                    errs.append('%s: "%s" cell %d,%d is "%s"' % (tag, key, i+1, j+1, c))
                    continue
                if c != 'N':
                    continue
                cell = rk.get('%d,%d' % (i + 1, j + 1))
                if not cell:
                    missing += 1
                    continue
                code = cell.get('code')
                if code not in CODES:
                    errs.append('%s: "%s" cell %d,%d has invalid code "%s"'
                                % (tag, key, i+1, j+1, code))
                # the citation must still exist verbatim in the characterisations
                a, b = ra[i], cb[j]
                hay = norm(chars.get(a, {}).get('requires', '') + ' ' +
                           chars.get(a, {}).get('provides', '') + ' ' +
                           chars.get(b, {}).get('requires', '') + ' ' +
                           chars.get(b, {}).get('provides', ''))
                for f in ('requires', 'blocked_by'):
                    q = norm(cell.get(f, ''))
                    if not q:
                        errs.append('%s: "%s" cell %d,%d missing "%s"'
                                    % (tag, key, i+1, j+1, f))
                    elif q not in hay:
                        errs.append('%s: "%s" cell %d,%d quotes "%s" which is not in '
                                    'either characterisation'
                                    % (tag, key, i+1, j+1, cell.get(f)[:48]))
    return errs, warns, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strict', action='store_true')
    a = ap.parse_args()

    ms = []
    for f in sorted(glob.glob(os.path.join(MAT, '*.json'))):
        ms += json.load(open(f, encoding='utf-8'))

    all_e, total_missing, n_cells = [], 0, 0
    per_matrix = []
    for i, m in enumerate(ms):
        e, w, miss = check(m, i)
        all_e += e
        total_missing += miss
        n = sum(r.count('N') for g in m.get('verdicts', {}).values() for r in g)
        n_cells += n
        per_matrix.append((m.get('domain', '?'), n - miss, n))

    print('=' * 70)
    print('  MATRIX VALIDATION against CRITERION.md v2')
    print('=' * 70)
    print('  matrices                 %d' % len(ms))
    print('  structural errors        %d' % len(all_e))
    print('  N cells total            %d' % n_cells)
    print('  N cells with a reason    %d  (%.1f%%)'
          % (n_cells - total_missing, 100.0 * (n_cells - total_missing) / max(n_cells, 1)))
    print('  N cells UNJUSTIFIED      %d' % total_missing)
    if all_e:
        print('\n  first errors:')
        for e in all_e[:12]:
            print('    - %s' % e)
    ok = not all_e and total_missing == 0
    print('\n  verdict: %s' % ('PASS' if ok else 'FAIL - corpus is not yet iron-bound'))
    if a.strict and not ok:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
