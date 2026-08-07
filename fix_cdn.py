with open('app2.py', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css',
    'https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css'
)
content = content.replace(
    'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js',
    'https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js'
)

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Switched Leaflet to jsdelivr CDN!')
