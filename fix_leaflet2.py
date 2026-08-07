with open('app2.py', encoding='utf-8') as f:
    content = f.read()

# Kontrolu guclendir: L.map var mi
content = content.replace(
    'if(typeof L===\"undefined\"){setTimeout(initMap,200);return;}',
    'if(typeof L===\"undefined\" || typeof L.map!==\"function\"){setTimeout(initMap,300);return;}'
)

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Stronger Leaflet check added!')
