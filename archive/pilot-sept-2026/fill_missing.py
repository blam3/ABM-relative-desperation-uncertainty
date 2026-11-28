import os
import csv, sys
sys.argv = ['x']
from pilot import worker
todo = {'legacy': [(4, 1, 2, 0.5), (7, 0, 1, 0.35)], 'fullfix': [(4, 1, 2, 0.35), (4, 1, 2, 0.5)]}
for st, items in todo.items():
    rows = [worker((st, r, b, a, t, 2, 'rank')) for r, b, a, t in items]
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results_{st}.csv'), 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writerows(rows)
    print('filled', st)
