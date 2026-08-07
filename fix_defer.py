with open('app2.py', encoding='utf-8') as f:
    content = f.read()

# updateMap cagrisini tarayici cizimi sonrasina ertele
old = '    renderResults(data);\n  });\n}'
new = '''    renderResults(data);
    setTimeout(function(){
      if(map){try{map.remove();}catch(e){} map=null; routeLayer=null; markerLayer=null;}
      initMap();
      updateMap(lastData);
    },400);
  });
}'''
content = content.replace(old, new)

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Deferred map rebuild added!')
