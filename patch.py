import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

MAP_FIX = """
// === BATIMETRIX MAP FIX ===
(function init() {
    if (typeof map === 'undefined' || typeof GUZERGAHLAR === 'undefined') {
        setTimeout(init, 300); return;
    }
    var _rl = null;
    window.redrawRoute = function(pts, analyzed) {
        if (_rl) { try { map.removeLayer(_rl); } catch(e){} }
        _rl = L.layerGroup().addTo(map);
        if (!pts || !pts.length) return;
        var coords = pts.map(function(p){ return [p.lat, p.lon]; });
        L.polyline(coords, {color:'#1ABC9C', weight:3, dashArray: analyzed ? '' : '8,5'}).addTo(_rl);
        pts.forEach(function(p, i) {
            var d = p.drag || 0.25;
            var col = d < 0.20 ? '#27AE60' : d < 0.35 ? '#F39C12' : '#E74C3C';
            L.circleMarker([p.lat, p.lon], {
                radius: (i===0||i===pts.length-1) ? 10 : 7,
                fillColor: col, color: '#fff', weight: 2, fillOpacity: 0.9
            }).bindPopup(
                '<div style="font-family:monospace"><b style="color:#1ABC9C">' +
                (p.isim||p.name||'WP') + '</b><br>' +
                (analyzed ?
                    'Drag: <b style="color:' + col + '">' + d.toFixed(4) + '</b><br>' +
                    'Depth: ' + (p.derinlik||p.depth||'?') + 'm' :
                    '<em style="color:#aaa">Preview</em>'
                ) + '</div>'
            ).addTo(_rl);
        });
        try { map.fitBounds(L.latLngBounds(coords), {padding:[40,40]}); } catch(e){}
        setTimeout(function(){ try { map.invalidateSize(); } catch(e){} }, 150);
    };

    var sel = document.getElementById('guzergah');
    if (sel) {
        sel.addEventListener('change', function() {
            redrawRoute(GUZERGAHLAR[this.value] || [], false);
        });
        redrawRoute(GUZERGAHLAR[sel.value] || [], false);
    }

    document.querySelectorAll('[data-tab]').forEach(function(btn) {
        btn.addEventListener('click', function() {
            setTimeout(function(){ try { map.invalidateSize(); } catch(e){} }, 200);
        });
    });

    setTimeout(function(){ try { map.invalidateSize(); } catch(e){} }, 800);
})();
"""

idx = content.rfind('</script>')
if idx > 0:
    content = content[:idx] + MAP_FIX + content[idx:]
    print("Patch 1: MAP_FIX OK")
else:
    print("HATA: </script> bulunamadi")

anchors = [
    'document.getElementById("results").style.display = "block"',
    "document.getElementById('results').style.display = 'block'",
    'document.getElementById("sonuclar").style.display = "block"',
    "document.getElementById('sonuclar').style.display = 'block'",
]
for anchor in anchors:
    if anchor in content:
        content = content.replace(anchor,
            anchor + '\n  if(typeof redrawRoute==="function"&&(data.noktalar||data.waypoints))redrawRoute(data.noktalar||data.waypoints,true);', 1)
        print("Patch 2: Analiz→harita guncelleme OK")
        break
else:
    print("Patch 2: anchor bulunamadi (skip)")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\nTAMAM! Simdi: python app.py")
