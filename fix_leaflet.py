with open('app2.py', encoding='utf-8') as f:
    content = f.read()

# initMap basina Leaflet yuklendi mi kontrolu ekle
old = 'function initMap(){\n  if(map) return;'
new = '''function initMap(){
  if(map) return;
  if(typeof L===\"undefined\"){setTimeout(initMap,200);return;}'''
content = content.replace(old, new)

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Leaflet load check added!')
