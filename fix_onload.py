with open('app2.py', encoding='utf-8') as f:
    content = f.read()

# Leaflet script'ine onload ekle - Leaflet hazir olunca haritayi baslat
old = '<script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>'
new = '<script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\" onload=\"window.leafletReady=true;if(window.pendingMapInit){initMap();}\"></script>'
content = content.replace(old, new)

# initMap kontrolunu leafletReady bayragina cevir
old2 = 'if(typeof L===\"undefined\" || typeof L.map!==\"function\"){setTimeout(initMap,300);return;}'
new2 = 'if(!window.leafletReady){window.pendingMapInit=true;return;}'
content = content.replace(old2, new2)

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Leaflet onload fix applied!')
