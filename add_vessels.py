# -*- coding: utf-8 -*-
with open('app2.py', encoding='utf-8') as f:
    content = f.read()

# (isim, a, c, dwt, fuel)
NEW = [
    ("Aframax Tanker",     5247.0, 0.610, 110000,  45),
    ("Suezmax Tanker",     5247.0, 0.610, 160000,  55),
    ("MR Product Tanker",  5247.0, 0.610,  50000,  25),
    ("ULCC Tanker",        5247.0, 0.610, 400000, 140),
    ("Supramax Bulk",      4745.0, 0.622,  58000,  28),
    ("Panamax Bulk",       4745.0, 0.622,  75000,  32),
    ("VLOC Valemax",       4745.0, 0.622, 400000,  65),
    ("ULCV Container",     1984.0, 0.489, 220000, 150),
    ("Feeder Container",   1984.0, 0.489,  20000,  22),
]

lines = content.split('\n')

# 1) CII_REF'e ekle
idx = None
for i, l in enumerate(lines):
    if l.startswith('CII_REF = {'):
        idx = i
        break
if idx is None:
    print('HATA: CII_REF bulunamadi!')
    raise SystemExit
cii = ['    "%s": {"a": %s, "c": %s},' % (n, a, c) for (n, a, c, d, f) in NEW]
lines[idx+1:idx+1] = cii

# 2) VESSEL_PROFILE'a ekle
idx = None
for i, l in enumerate(lines):
    if l.startswith('VESSEL_PROFILE = {'):
        idx = i
        break
if idx is None:
    print('HATA: VESSEL_PROFILE bulunamadi!')
    raise SystemExit
prof = ['    "%s": {"dwt":%d,"fuel":%d},' % (n, d, f) for (n, a, c, d, f) in NEW]
lines[idx+1:idx+1] = prof

# 3) Dropdown'a ekle
idx = None
for i, l in enumerate(lines):
    if 'value="VLCC Tanker"' in l:
        idx = i
        break
if idx is None:
    print('HATA: dropdown bulunamadi!')
    raise SystemExit
opts = ['        <option value="%s">%s</option>' % (n, n) for (n, a, c, d, f) in NEW]
lines[idx+1:idx+1] = opts

content = '\n'.join(lines)
with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('%d vessel types added! Total: %d' % (len(NEW), len(NEW)+6))