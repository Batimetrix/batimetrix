p = 'README.md'
s = open(p, encoding='utf-8-sig').read()
a = 'Details and limits: see [Early Validation](#early-validation-real-ship-data).'
b = '- ESA (2015). ESA helps faster, cleaner shipping (Globcurrent project).'
print('OK link' if a in s else 'MISS link'); print('OK esa' if b in s else 'MISS esa')
s = s.replace(a, a + '\n\n**Technical note (draft, not peer reviewed):** '
              '[Batimetrix Technical Note 1 (PDF)](docs/Batimetrix_Technical_Note_1_v2.pdf)', 1)
s = s.replace(b, '- ESA. ESA helps faster, cleaner shipping (Globcurrent project).', 1)
open(p, 'w', encoding='utf-8').write(s)
