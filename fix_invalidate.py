with open('app2.py', encoding='utf-8') as f:
    content = f.read()

# updateMap sonunda haritayi yeniden boyutlandir
old = '''  markerLayer.addTo(map);
  map.fitBounds(routeLayer.getBounds(),{padding:[30,30]});
}'''
new = '''  markerLayer.addTo(map);
  setTimeout(function(){
    map.invalidateSize();
    map.fitBounds(routeLayer.getBounds(),{padding:[30,30]});
  },300);
}'''
content = content.replace(old, new)

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('invalidateSize added to updateMap!')
