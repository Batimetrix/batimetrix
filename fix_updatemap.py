with open('app2.py', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'function updateMap(',
    'function updateMap(_XX_'
).replace(
    'function updateMap(_XX_',
    'function updateMap('
)

# updateMap basina kontrol ekle - once updateMap tanimini bulalim
import re
content = content.replace(
    'function updateMap(data){',
    'function updateMap(data){\n  if(typeof L===\"undefined\"||typeof L.polyline!==\"function\"){setTimeout(function(){updateMap(data);},300);return;}'
)

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('updateMap Leaflet check added!')
