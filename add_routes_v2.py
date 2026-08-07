# -*- coding: utf-8 -*-
with open('app2.py', encoding='utf-8') as f:
    content = f.read()

new_routes = {
    # --- KURU YUK / DRY BULK ---
    "porthedland_qingdao": ("Port Hedland \u2192 Qingdao (Iron Ore)", [
        ("Port Hedland",-20.31,118.58,15,0.06),("Timor Sea",-11.00,123.00,2500,0.14),
        ("Makassar Strait",-2.00,118.00,2000,0.12),("Philippine Sea",10.00,125.00,4500,0.20),
        ("Taiwan Strait",24.00,119.00,60,0.12),("Qingdao",36.07,120.38,20,0.06)]),
    "pontamadeira_qingdao": ("Ponta da Madeira \u2192 Qingdao (Vale Ore)", [
        ("Ponta da Madeira",-2.56,-44.37,25,0.08),("Equator Atlantic",0.00,-30.00,4200,0.16),
        ("Cape of Good Hope",-35.00,20.00,3000,0.28),("Indian Ocean",-25.00,60.00,4500,0.22),
        ("Malacca Strait",2.50,101.00,40,0.10),("Qingdao",36.07,120.38,20,0.06)]),
    "newcastle_tokyo": ("Newcastle \u2192 Tokyo (Coal)", [
        ("Newcastle AU",-32.92,151.78,15,0.10),("Coral Sea",-20.00,155.00,3500,0.20),
        ("Philippine Sea",10.00,140.00,5500,0.24),("Tokyo",35.65,139.84,20,0.07)]),
    "neworleans_shanghai": ("New Orleans \u2192 Shanghai (Grain)", [
        ("New Orleans",29.95,-90.07,15,0.05),("Gulf of Mexico",25.00,-88.00,2000,0.14),
        ("Panama Canal",9.08,-79.68,20,0.06),("Pacific Central",15.00,-110.00,3800,0.20),
        ("Mid-Pacific",25.00,-160.00,5500,0.32),("Shanghai",31.23,121.47,15,0.06)]),

    # --- ENERJI / ENERGY ---
    "rastanura_ningbo": ("Ras Tanura \u2192 Ningbo (VLCC Crude)", [
        ("Ras Tanura",26.64,50.16,20,0.04),("Hormuz Strait",26.57,56.25,60,0.08),
        ("Arabian Sea",18.00,62.00,3500,0.16),("Indian Ocean",5.00,80.00,3800,0.18),
        ("Malacca Strait",2.50,101.00,40,0.10),("Ningbo",29.87,121.55,20,0.06)]),
    "raslaffan_tokyo": ("Ras Laffan \u2192 Tokyo (LNG)", [
        ("Ras Laffan",25.92,51.55,20,0.04),("Hormuz Strait",26.57,56.25,60,0.08),
        ("Arabian Sea",18.00,62.00,3500,0.16),("Malacca Strait",2.50,101.00,40,0.10),
        ("South China Sea",12.00,115.00,4000,0.15),("Tokyo",35.65,139.84,20,0.07)]),
    "sabinepass_rotterdam": ("Sabine Pass \u2192 Rotterdam (US LNG)", [
        ("Sabine Pass",29.73,-93.87,15,0.05),("Gulf of Mexico",26.00,-88.00,2000,0.14),
        ("Florida Strait",24.50,-80.00,800,0.16),("North Atlantic",40.00,-45.00,4500,0.30),
        ("Rotterdam",51.95,4.14,25,0.04)]),
    "bonny_rotterdam": ("Bonny \u2192 Rotterdam (Crude)", [
        ("Bonny",4.45,7.17,20,0.07),("Gulf of Guinea",2.00,3.00,3000,0.14),
        ("Equator Atlantic",0.00,-5.00,4000,0.16),("Canary Islands",28.00,-18.00,3500,0.15),
        ("Rotterdam",51.95,4.14,25,0.04)]),

    # --- 2026 JEOPOLITIK ---
    "singapore_rotterdam_cape": ("Singapore \u2192 Rotterdam (Cape Route)", [
        ("Singapore",1.29,103.85,25,0.08),("Indian Ocean",-5.00,80.00,4000,0.18),
        ("Cape of Good Hope",-35.00,20.00,3000,0.30),("South Atlantic",-15.00,-5.00,4500,0.22),
        ("Canary Islands",28.00,-18.00,3500,0.15),("Rotterdam",51.95,4.14,25,0.04)]),
    "murmansk_shanghai": ("Murmansk \u2192 Shanghai (Arctic NSR)", [
        ("Murmansk",68.97,33.05,30,0.06),("Kara Sea",75.00,65.00,150,0.10),
        ("Laptev Sea",76.00,125.00,50,0.09),("Bering Strait",65.80,-169.00,50,0.12),
        ("Shanghai",31.23,121.47,15,0.06)]),
    "hormuz_transit": ("Kuwait \u2192 Arabian Sea (Hormuz)", [
        ("Kuwait",29.07,48.13,15,0.04),("Persian Gulf",27.00,52.00,60,0.06),
        ("Hormuz Strait",26.57,56.25,60,0.08),("Gulf of Oman",24.00,58.00,2000,0.12),
        ("Arabian Sea",20.00,62.00,3500,0.16)]),

    # --- BOGAZLAR ---
    "dover_strait": ("Rotterdam \u2192 Felixstowe (Dover)", [
        ("Rotterdam",51.95,4.14,25,0.04),("North Sea",52.50,3.00,30,0.06),
        ("Dover Strait",51.00,1.50,35,0.07),("Felixstowe",51.96,1.31,20,0.05)]),
    "babelmandeb": ("Jeddah \u2192 Djibouti (Bab el-Mandeb)", [
        ("Jeddah",21.49,39.19,30,0.06),("Red Sea South",17.00,41.00,1500,0.11),
        ("Bab el-Mandeb",12.60,43.30,180,0.09),("Djibouti",11.60,43.15,20,0.06)]),

    # --- COGRAFI BOSLUKLAR ---
    "tangermed_rotterdam": ("Tanger Med \u2192 Rotterdam", [
        ("Tanger Med",35.89,-5.50,20,0.07),("Gibraltar",36.14,-5.35,300,0.08),
        ("Atlantic Iberia",40.00,-10.00,3000,0.18),("Biscay",45.00,-5.00,2800,0.25),
        ("Rotterdam",51.95,4.14,25,0.04)]),
    "callao_shanghai": ("Callao \u2192 Shanghai (Copper)", [
        ("Callao",-12.05,-77.15,20,0.08),("Pacific South",-10.00,-100.00,4000,0.20),
        ("Mid-Pacific",0.00,-140.00,4800,0.24),("Philippine Sea",15.00,135.00,5000,0.22),
        ("Shanghai",31.23,121.47,15,0.06)]),
    "lagos_rotterdam": ("Lagos \u2192 Rotterdam", [
        ("Lagos",6.44,3.40,15,0.07),("Gulf of Guinea",2.00,0.00,3000,0.14),
        ("Equator Atlantic",5.00,-15.00,4000,0.16),("Canary Islands",28.00,-18.00,3500,0.15),
        ("Rotterdam",51.95,4.14,25,0.04)]),
    "gdansk_rotterdam": ("Gdansk \u2192 Rotterdam (Baltic)", [
        ("Gdansk",54.40,18.66,15,0.04),("Baltic Sea",55.00,15.00,80,0.06),
        ("Danish Straits",55.50,11.00,30,0.05),("North Sea",55.00,5.00,40,0.08),
        ("Rotterdam",51.95,4.14,25,0.04)]),
    "hochiminh_losangeles": ("Ho Chi Minh \u2192 Los Angeles", [
        ("Ho Chi Minh",10.77,106.70,15,0.08),("South China Sea",12.00,112.00,4000,0.15),
        ("Philippine Sea",18.00,130.00,5000,0.22),("Mid-Pacific",30.00,-175.00,5800,0.34),
        ("Los Angeles",33.74,-118.27,20,0.06)]),
    "chittagong_singapore": ("Chittagong \u2192 Singapore", [
        ("Chittagong",22.30,91.80,10,0.07),("Bay of Bengal",15.00,90.00,3000,0.16),
        ("Andaman Sea",8.00,96.00,1500,0.12),("Malacca Strait",2.50,101.00,40,0.10),
        ("Singapore",1.29,103.85,25,0.08)]),
    "manzanillo_shanghai": ("Manzanillo \u2192 Shanghai", [
        ("Manzanillo",19.05,-104.32,20,0.06),("Pacific East",18.00,-120.00,3500,0.20),
        ("Mid-Pacific",25.00,-160.00,5500,0.32),("East China Sea",29.00,123.00,100,0.10),
        ("Shanghai",31.23,121.47,15,0.06)]),
}

# --- 1) ROUTES sozlugune ekle ---
entries = ""
for key,(name,wps) in new_routes.items():
    entries += '    "'+key+'": {\n        "name": "'+name+'",\n        "waypoints": [\n'
    for w in wps:
        entries += '            {"name":"'+w[0]+'","lat":'+str(w[1])+',"lon":'+str(w[2])+',"depth":'+str(w[3])+',"ssh":'+str(w[4])+'},\n'
    entries += '        ]\n    },\n'

anchor = '        ]\n    },\n}'
if content.count(anchor) != 1:
    print('HATA: anchor bulunamadi veya birden fazla! count =', content.count(anchor))
    raise SystemExit
content = content.replace(anchor, '        ]\n    },\n' + entries + '}')

# --- 2) Dropdown'a ekle ---
lines = content.split('\n')
idx = None
for i,l in enumerate(lines):
    if 'value="genoa_alexandria"' in l:
        idx = i
        break
if idx is None:
    print('HATA: dropdown anchor bulunamadi!')
    raise SystemExit

opts = []
for key,(name,wps) in new_routes.items():
    opts.append('        <option value="'+key+'">'+name+'</option>')
lines[idx+1:idx+1] = opts
content = '\n'.join(lines)

with open('app2.py','w',encoding='utf-8') as f:
    f.write(content)
print(str(len(new_routes))+' routes added! Total: '+str(len(new_routes)+40))