# -*- coding: utf-8 -*-
with open('app2.py', encoding='utf-8') as f:
    content = f.read()

# 1) CSS ekle - kayan saat seridi stilleri
css_marker = '/* SAT BAR */'
css_new = '''/* WORLD CLOCK TICKER */
.wct-wrap{background:#050F1C;border-bottom:1px solid var(--line);overflow:hidden;white-space:nowrap;padding:5px 0;position:relative}
.wct-track{display:inline-block;white-space:nowrap;animation:wctScroll 90s linear infinite}
.wct-wrap:hover .wct-track{animation-play-state:paused}
.wct-item{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--mute);margin:0 14px;letter-spacing:.5px}
.wct-item b{color:var(--teal);font-weight:600}
.wct-item.wct-hl b{color:#FFC107}
.wct-item .wct-off{color:#2C3E50;font-size:9px;margin-left:4px}
@keyframes wctScroll{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
/* SAT BAR */'''
content = content.replace(css_marker, css_new, 1)

# 2) HTML ekle - header kapanisindan sonra serit
html_marker = '</header>'
html_new = '''</header>
<div class="wct-wrap"><div class="wct-track" id="wct_track"></div></div>'''
content = content.replace(html_marker, html_new, 1)

# 3) JS ekle - 38 dilim, window.onload oncesine
js_marker = '// Init on load'
js_new = '''// ===== WORLD CLOCKS (38 UTC offsets) =====
var WCT_ZONES=[
[-12,"BAKER IS."],[-11,"PAGO PAGO"],[-10,"HONOLULU"],[-9.5,"MARQUESAS"],[-9,"ANCHORAGE"],
[-8,"LOS ANGELES"],[-7,"DENVER"],[-6,"MEXICO CITY"],[-5,"NEW YORK"],[-4,"SANTIAGO"],
[-3.5,"ST. JOHN'S"],[-3,"SAO PAULO"],[-2,"S. GEORGIA"],[-1,"AZORES"],
[0,"LONDON"],[1,"PARIS"],[2,"CAIRO"],[3,"ISTANBUL"],[3.5,"TEHRAN"],[4,"DUBAI"],
[4.5,"KABUL"],[5,"KARACHI"],[5.5,"MUMBAI"],[5.75,"KATHMANDU"],[6,"DHAKA"],
[6.5,"YANGON"],[7,"BANGKOK"],[8,"SHANGHAI"],[8.75,"EUCLA"],[9,"TOKYO"],
[9.5,"ADELAIDE"],[10,"SYDNEY"],[10.5,"LORD HOWE"],[11,"HONIARA"],[12,"AUCKLAND"],
[12.75,"CHATHAM"],[13,"NUKUALOFA"],[14,"KIRITIMATI"]
];
function wctFmt(off){
  var h=Math.floor(Math.abs(off)),m=Math.round((Math.abs(off)-h)*60);
  return (off<0?"-":"+")+h+(m?":"+(m<10?"0":"")+m:"");
}
function wctRender(){
  var now=new Date();
  var utcMs=now.getTime()+now.getTimezoneOffset()*60000;
  var html="";
  WCT_ZONES.forEach(function(z){
    var t=new Date(utcMs+z[0]*3600000);
    var hh=("0"+t.getHours()).slice(-2),mm=("0"+t.getMinutes()).slice(-2);
    var hl=z[1]==="ISTANBUL"?" wct-hl":"";
    html+='<span class="wct-item'+hl+'">'+z[1]+' <b>'+hh+":"+mm+'</b><span class="wct-off">UTC'+wctFmt(z[0])+'</span></span>';
  });
  var el=document.getElementById("wct_track");
  if(el) el.innerHTML=html+html;
}
setInterval(wctRender,30000);
wctRender();

// Init on load'''
content = content.replace(js_marker, js_new, 1)

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('38 world clocks ticker added!')