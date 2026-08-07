# -*- coding: utf-8 -*-
import re, shutil

SRC = 'app2.py'
shutil.copy(SRC, 'app2_backup.py')
print('Backup created: app2_backup.py')

with open(SRC, encoding='utf-8') as f:
    c = f.read()

log = []

# ===== 1) VESSEL_PROFILE : fuel / dwt kalibrasyonu =====
PROFILE_FIX = {
    "VLCC Tanker":       {"fuel": 75},
    "ULCC Tanker":       {"fuel": 95},
    "MR Product Tanker": {"fuel": 32},
    "Supramax Bulk":     {"fuel": 32},
    "LNG Carrier":       {"dwt": 100000},
}
s = c.index('VESSEL_PROFILE = {')
e = c.index('\n}', s) + 2
blk = c[s:e]
for name, ch in PROFILE_FIX.items():
    m = re.search(r'"' + re.escape(name) + r'"\s*:\s*\{([^}]*)\}', blk)
    if not m:
        log.append('MISS  profile: ' + name); continue
    body = m.group(1)
    for k, v in ch.items():
        body = re.sub(r'"' + k + r'"\s*:\s*\d+', '"%s":%d' % (k, v), body)
    blk = blk[:m.start(1)] + body + blk[m.end(1):]
    log.append('OK    profile: %-18s %s' % (name, ch))
c = c[:s] + blk + c[e:]

# ===== 2) CII_REF : gemi tipine gore A/B/C/D/E sinirlari =====
TNK = '[0.82,0.93,1.08,1.28]'   # Tanker
BLK = '[0.86,0.94,1.06,1.18]'   # Bulk carrier
BOX = '[0.83,0.94,1.07,1.19]'   # Container
GEN = '[0.83,0.94,1.06,1.19]'   # General cargo
LNG = '[0.89,0.98,1.06,1.13]'   # LNG >= 100,000 DWT
BOUNDS = {
    "VLCC Tanker": TNK, "ULCC Tanker": TNK, "Aframax Tanker": TNK,
    "Suezmax Tanker": TNK, "MR Product Tanker": TNK,
    "Capesize Bulk": BLK, "Handy Bulk": BLK, "Supramax Bulk": BLK,
    "Panamax Bulk": BLK, "VLOC Valemax": BLK,
    "Panamax Container": BOX, "ULCV Container": BOX, "Feeder Container": BOX,
    "LNG Carrier": LNG,
    "Black Sea Cargo": GEN,
}
CAP = {"VLOC Valemax": 279000}   # IMO: bulk >= 279,000 DWT tavani

s = c.index('CII_REF = {')
e = c.index('\n}', s) + 2
blk = c[s:e]
for name, d in BOUNDS.items():
    m = re.search(r'"' + re.escape(name) + r'"\s*:\s*\{([^}]*)\}', blk)
    if not m:
        log.append('MISS  cii_ref: ' + name); continue
    body = m.group(1).rstrip().rstrip(',')
    if '"d"' in body:
        log.append('SKIP  cii_ref: ' + name); continue
    body += ', "d": ' + d
    if name in CAP:
        body += ', "cap": %d' % CAP[name]
    blk = blk[:m.start(1)] + body + blk[m.end(1):]
    log.append('OK    cii_ref: ' + name)
cii_end = s + len(blk)
c = c[:s] + blk + c[e:]

# ===== 3) fallback'e de sinir ekle =====
old_fb = 'CII_REF.get(vessel,{"a":588.0,"c":0.3885})'
new_fb = 'CII_REF.get(vessel,{"a":588.0,"c":0.3885,"d":[0.83,0.94,1.06,1.19]})'
if old_fb in c:
    c = c.replace(old_fb, new_fb, 1); log.append('OK    fallback boundaries')
else:
    log.append('MISS  fallback')

# ===== 4) kapasite tavani (VLOC) =====
old_req = 'req = p["a"] * (dwt**(-p["c"])) * (1-11/100)'
new_req = 'cap = min(dwt, p.get("cap", dwt))\n    req = p["a"] * (cap**(-p["c"])) * (1-11/100)'
if old_req in c:
    c = c.replace(old_req, new_req, 1); log.append('OK    capacity cap (279k)')
else:
    log.append('MISS  req line')

# ===== 5) not verme fonksiyonunu bul ve degistir =====
idx = c.find('0.86', cii_end)
if idx == -1:
    log.append('MISS  grade function (0.86 yok)')
else:
    ds = c.rfind('def ', cii_end, idx)
    ee = c.find('return "E"', idx)
    if ds == -1 or ee == -1:
        log.append('MISS  grade function sinirlari')
    else:
        fn_end = ee + len('return "E"')
        old_fn = c[ds:fn_end]
        print('\n--- BULUNAN FONKSIYON ---\n' + old_fn + '\n-------------------------\n')
        mm = re.match(r'def\s+(\w+)\s*\(', old_fn)
        fn = mm.group(1)
        ls = c.rfind('\n', 0, ds) + 1
        ind = c[ls:ds]
        i2 = ind + '    '
        new_fn = ('def %s(v, r, d=None):\n' % fn +
                  i2 + 'd = d or [0.86,0.94,1.06,1.18]\n' +
                  i2 + 'o = v / r\n' +
                  i2 + 'if o <= d[0]: return "A"\n' +
                  i2 + 'if o <= d[1]: return "B"\n' +
                  i2 + 'if o <= d[2]: return "C"\n' +
                  i2 + 'if o <= d[3]: return "D"\n' +
                  i2 + 'return "E"')
        c = c[:ds] + new_fn + c[fn_end:]
        log.append('OK    grade function: ' + fn)
        pat = re.compile(r'\b' + fn + r'\s*\(\s*([\w\.]+)\s*,\s*([\w\.]+)\s*\)')
        n = len(pat.findall(c))
        c = pat.sub(lambda m: '%s(%s, %s, p.get("d"))' % (fn, m.group(1), m.group(2)), c)
        log.append('OK    call sites patched: %d' % n)

# ===== yaz =====
print('\n'.join(log))
if any(l.startswith('MISS') for l in log):
    print('\n!!! MISS var - dosya YAZILMADI. Yukaridaki raporu paylas.')
else:
    with open(SRC, 'w', encoding='utf-8') as f:
        f.write(c)
    print('\napp2.py updated!')