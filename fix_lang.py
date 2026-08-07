with open('app2.py', encoding='utf-8') as f:
    content = f.read()

# Dil sozlugunu LANG olarak yeniden adlandir - Leaflet'in L'sini ezmesin
content = content.replace('var L={', 'var LANG={')
content = content.replace('var t=L[l];', 'var t=LANG[l];')
content = content.replace('var t=L[lang];', 'var t=LANG[lang];')

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Language dict renamed L -> LANG! Leaflet is free!')
