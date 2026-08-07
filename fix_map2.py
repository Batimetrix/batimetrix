with open('app2.py', encoding='utf-8') as f:
    content = f.read()

# Sayfa yuklenince haritayi zorla baslat
old = 'if(!map && id===\"map_tab\"){initMap();} if(map) setTimeout(function(){map.invalidateSize();},100);'
new = 'if(id===\"map_tab\"){if(!map){initMap();} setTimeout(function(){if(map)map.invalidateSize();},200);}'
content = content.replace(old, new)

# Ayrica sayfa aciliminda otomatik baslat
if 'window.addEventListener(\"load\"' not in content:
    content = content.replace(
        'initMap();\n</script>',
        'setTimeout(function(){if(!map)initMap();setTimeout(function(){if(map)map.invalidateSize();},300);},500);\n</script>'
    )

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Map init forced on load')
