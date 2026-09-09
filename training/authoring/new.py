# -*- coding: utf-8 -*-
"""Creates new v2 source matrices from a compact text spec on stdin.

    DOMAIN: urban flooding
    GROUP: hazard
    PARAM: SOURCE
    - Value name :: requires text :: provides text
    PARAM: RESPONSE
    - ...
    EXCL: SOURCE|RESPONSE
    Row value >< Col value | N3 | requires quote | blocked_by quote
    ---                          <- separates matrices

Citations are checked verbatim as they are written, so a bad quote is caught
here rather than at build time.
"""
import json, os, re, sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')


def norm(s):
    return ' '.join((s or '').lower().split())


def parse(block, errs):
    m = {'domain': '', 'group': '', 'parameters': {}, 'characterisations': {},
         'exclusions': {}}
    cur_param = cur_excl = None
    for raw in block.split('\n'):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.upper().startswith('DOMAIN:'):
            m['domain'] = line.split(':', 1)[1].strip(); continue
        if line.upper().startswith('GROUP:'):
            m['group'] = line.split(':', 1)[1].strip(); continue
        if line.upper().startswith('PARAM:'):
            cur_param = line.split(':', 1)[1].strip()
            m['parameters'][cur_param] = []; cur_excl = None; continue
        if line.upper().startswith('EXCL:'):
            cur_excl = line.split(':', 1)[1].strip()
            m['exclusions'].setdefault(cur_excl, []); cur_param = None; continue
        if line.startswith('-') and cur_param:
            parts = [x.strip() for x in line[1:].split('::')]
            if len(parts) != 3:
                errs.append('%s: bad value line: %s' % (m['domain'], line[:60])); continue
            v, r, p = parts
            if v in m['characterisations']:
                errs.append('%s: duplicate value "%s"' % (m['domain'], v)); continue
            m['parameters'][cur_param].append(v)
            m['characterisations'][v] = {'requires': r, 'provides': p}
            continue
        if '><' in line and cur_excl:
            try:
                lhs, code, req, blk = [x.strip() for x in line.split('|')]
                row, col = [x.strip() for x in lhs.split('><')]
            except ValueError:
                errs.append('%s: bad exclusion: %s' % (m['domain'], line[:60])); continue
            m['exclusions'][cur_excl].append(
                {'row': row, 'col': col, 'code': code, 'requires': req, 'blocked_by': blk})
            continue
        errs.append('%s: unrecognised line: %s' % (m['domain'], line[:60]))
    return m


def check(m, errs):
    ch, params = m['characterisations'], m['parameters']
    if not m['domain']:
        errs.append('a matrix has no DOMAIN')
    for p, vs in params.items():
        if len(vs) < 3:
            errs.append('%s: parameter "%s" has %d values' % (m['domain'], p, len(vs)))
    for key, lst in m['exclusions'].items():
        if '|' not in key:
            errs.append('%s: bad EXCL key "%s"' % (m['domain'], key)); continue
        pa, pb = key.split('|')
        if pa not in params or pb not in params:
            errs.append('%s: EXCL "%s" names an unknown parameter' % (m['domain'], key)); continue
        # build_matrices.py derives pair keys with combinations() over the parameters in
        # DECLARATION order, so "B|A" is a key it will never generate and the exclusions
        # under it would be silently dropped. Catch it here instead.
        names = list(params)
        if names.index(pa) > names.index(pb):
            errs.append('%s: EXCL "%s" is reversed -- write it as "%s|%s" and swap row/col'
                        % (m['domain'], key, pb, pa)); continue
        seen = set()
        for e in lst:
            sig = (e.get('row'), e.get('col'))
            if sig in seen:
                errs.append('%s [%s]: duplicate exclusion %s x %s' % (m['domain'], key, sig[0], sig[1]))
            seen.add(sig)
        for e in lst:
            if e['row'] not in params[pa]:
                errs.append('%s [%s]: "%s" not a value of %s'
                            % (m['domain'], key, e['row'], pa)); continue
            if e['col'] not in params[pb]:
                errs.append('%s [%s]: "%s" not a value of %s'
                            % (m['domain'], key, e['col'], pb)); continue
            hay = norm(' '.join([ch[e['row']]['requires'], ch[e['row']]['provides'],
                                 ch[e['col']]['requires'], ch[e['col']]['provides']]))
            for f in ('requires', 'blocked_by'):
                if norm(e[f]) not in hay:
                    errs.append('%s [%s]: %s x %s quote not found: "%s"'
                                % (m['domain'], key, e['row'], e['col'], e[f][:46]))


def main():
    start = int(sys.argv[1])
    errs, mats = [], []
    for block in sys.stdin.read().split('\n---\n'):
        if not block.strip():
            continue
        m = parse(block, errs)
        check(m, errs)
        mats.append(m)
    if errs:
        print('  %d ERRORS - nothing written' % len(errs))
        for e in errs[:20]:
            print('    - %s' % e)
        return 1
    for i, m in enumerate(mats):
        n = start + i
        slug = '-'.join(x for x in re.sub(r'[^a-z0-9]+', '-', m['domain'].lower()).split('-') if x)
        path = os.path.join(SRC, '%03d-%s.json' % (n, slug))
        json.dump(m, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
        ex = sum(len(v) for v in m['exclusions'].values())
        print('  %03d %-44s %2d params  %2d exclusions' % (n, m['domain'], len(m['parameters']), ex))
    print('  %d matrices written' % len(mats))
    return 0


if __name__ == '__main__':
    sys.exit(main())
