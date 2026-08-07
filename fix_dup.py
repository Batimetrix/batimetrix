with open('app2.py', encoding='utf-8') as f:
    content = f.read()

# renderResults icindeki erken updateMap cagrisini kaldir (geciktirilmis olani kaliyor)
content = content.replace('\n  updateMap(data);\n', '\n', 1)

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Dogrula
lines = content.split(chr(10))
count = sum(1 for l in lines if 'updateMap(' in l and 'function' not in l and 'setTimeout(function(){updateMap' not in l)
print('Early call removed! Remaining updateMap calls:', count)
