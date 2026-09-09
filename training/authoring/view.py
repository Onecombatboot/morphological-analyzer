# -*- coding: utf-8 -*-
"""Prints one source file's candidates with the evidence needed to re-decide them.
   python authoring/view.py 001"""
import json, glob, os, sys
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
f = glob.glob(os.path.join(SRC, sys.argv[1] + '-*.json'))[0]
m = json.load(open(f, encoding='utf-8'))
ch = m['characterisations']
print('FILE %s' % os.path.basename(f))
print('DOMAIN %s' % m['domain'])
print('PARAMS')
for p, vs in m['parameters'].items():
    print('  %s' % p)
    for v in vs:
        print('    %-42s R: %s' % (v, ch[v]['requires']))
        print('    %-42s P: %s' % ('', ch[v]['provides']))
print('CANDIDATES (%d)' % sum(len(v) for v in m['candidates'].values()))
for k, lst in m['candidates'].items():
    print('  %s' % k)
    for e in lst:
        print('    %-40s x %s' % (e['row'], e['col']))
