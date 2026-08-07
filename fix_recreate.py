with open('app2.py', encoding='utf-8') as f:
    content = f.read()

# updateMap icinde haritayi yik ve yeniden yarat
old = '''  if(typeof L==="undefined"||typeof L.polyline!=="function"){setTimeout(function(){updateMap(data);},300);return;}
  initMap();'''
new = '''  if(typeof L==="undefined"||typeof L.polyline!=="function"){setTimeout(function(){updateMap(data);},300);return;}
  if(map){try{map.remove();}catch(e){} map=null; routeLayer=null; markerLayer=null;}
  initMap();'''
content = content.replace(old, new)

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Map destroy-recreate added to updateMap!')
