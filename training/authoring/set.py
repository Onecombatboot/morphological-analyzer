# -*- coding: utf-8 -*-
"""Writes exclusions into a source file from a compact spec on stdin.

  PARAM A|PARAM B
  row value >< col value | CODE | requires quote | blocked_by quote

Citations are checked verbatim against the characterisations as they are written,
so a bad quote is caught here rather than at build time. Any candidate not listed
becomes Y -- that is the point of the format.

  python authoring/set.py 001 < spec.txt
"""
import json, glob, os, sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')


def norm(s):
    return ' '.join((s or '').lower().split())


def main():
    f = glob.glob(os.path.join(SRC, sys.argv[1] + '-*.json'))[0]
    m = json.load(open(f, encoding='utf-8'))
    ch, params = m['characterisations'], m['parameters']

    excl, key, errs, n = {}, None, [], 0
    for raw in sys.stdin.read().split('\n'):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if '><' not in line:
            key = line
            if key not in ['%s|%s' % (a, b) for a in params for b in params]:
                errs.append('unknown parameter pair: %s' % key)
            excl.setdefault(key, [])
            continue
        try:
            lhs, code, req, blk = [x.strip() for x in line.split('|')]
            row, col = [x.strip() for x in lhs.split('><')]
        except ValueError:
            errs.append('cannot parse: %s' % line[:70]); continue
        pa, pb = key.split('|')
        if row not in params.get(pa, []):
            errs.append('%s: "%s" is not a value of %s' % (key, row, pa)); continue
        if col not in params.get(pb, []):
            errs.append('%s: "%s" is not a value of %s' % (key, col, pb)); continue
        hay = norm(' '.join([ch[row]['requires'], ch[row]['provides'],
                             ch[col]['requires'], ch[col]['provides']]))
        for q in (req, blk):
            if norm(q) not in hay:
                errs.append('%s: %s x %s quote not found: "%s"' % (key, row, col, q[:50]))
        excl[key].append({'row': row, 'col': col, 'code': code,
                          'requires': req, 'blocked_by': blk})
        n += 1

    if errs:
        print('  %d ERRORS - nothing written' % len(errs))
        for e in errs[:15]:
            print('    - %s' % e)
        return 1

    m['exclusions'] = {k: v for k, v in excl.items() if v}
    cand = sum(len(v) for v in m.get('candidates', {}).values())
    m.pop('candidates', None)
    json.dump(m, open(f, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('  %s: %d exclusions written (was %d v1 candidates)'
          % (os.path.basename(f), n, cand))
    return 0


if __name__ == '__main__':
    sys.exit(main())
