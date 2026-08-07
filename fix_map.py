with open('app2.py', encoding='utf-8') as f:
    content = f.read()

# showTab icinde map yoksa olustur
old = 'if(map) setTimeout(function(){map.invalidateSize();},100);'
new = 'if(!map && id===\"map_tab\"){initMap();} if(map) setTimeout(function(){map.invalidateSize();},100);'
content = content.replace(old, new)

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Map auto-init fixed!')
