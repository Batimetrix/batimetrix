# -*- coding: utf-8 -*-
with open('app2.py', encoding='utf-8') as f:
    content = f.read()

# 34 yeni global ticaret rotasi (mevcut 6 + bunlar = 40)
new_routes = {
    "shanghai_rotterdam": ("Shanghai \u2192 Rotterdam", [
        ("Shanghai",31.23,121.47,15,0.06),("Malacca Strait",2.50,101.00,40,0.10),
        ("Indian Ocean",5.00,75.00,3800,0.20),("Suez",29.97,32.55,20,0.05),
        ("Mediterranean",36.00,15.00,2500,0.12),("Rotterdam",51.95,4.14,25,0.04)]),
    "shanghai_losangeles": ("Shanghai \u2192 Los Angeles", [
        ("Shanghai",31.23,121.47,15,0.06),("Pacific West",35.00,150.00,5500,0.30),
        ("Mid-Pacific",38.00,180.00,5800,0.35),("Pacific East",35.00,-140.00,4500,0.28),
        ("Los Angeles",33.74,-118.27,20,0.06)]),
    "singapore_shanghai": ("Singapore \u2192 Shanghai", [
        ("Singapore",1.29,103.85,25,0.08),("South China Sea",10.00,110.00,4000,0.15),
        ("Taiwan Strait",24.00,119.00,60,0.12),("Shanghai",31.23,121.47,15,0.06)]),
    "rotterdam_newyork": ("Rotterdam \u2192 New York", [
        ("Rotterdam",51.95,4.14,25,0.04),("English Channel",50.00,-1.00,50,0.10),
        ("North Atlantic",48.00,-30.00,4200,0.35),("Grand Banks",44.00,-50.00,100,0.25),
        ("New York",40.60,-74.05,20,0.06)]),
    "santos_rotterdam": ("Santos \u2192 Rotterdam", [
        ("Santos",-23.98,-46.30,15,0.07),("South Atlantic",-15.00,-30.00,4500,0.22),
        ("Equator Atlantic",0.00,-25.00,4000,0.18),("Canary Islands",28.00,-18.00,3500,0.15),
        ("Rotterdam",51.95,4.14,25,0.04)]),
    "dubai_singapore": ("Dubai \u2192 Singapore", [
        ("Dubai",25.27,55.30,20,0.05),("Arabian Sea",18.00,62.00,3500,0.16),
        ("Indian Ocean",8.00,75.00,3800,0.18),("Malacca Strait",2.50,101.00,40,0.10),
        ("Singapore",1.29,103.85,25,0.08)]),
    "mumbai_suez": ("Mumbai \u2192 Suez", [
        ("Mumbai",18.94,72.83,15,0.06),("Arabian Sea",18.00,62.00,3500,0.16),
        ("Gulf of Aden",12.50,45.00,2000,0.14),("Red Sea",20.00,38.00,1800,0.12),
        ("Suez",29.97,32.55,20,0.05)]),
    "durban_singapore": ("Durban \u2192 Singapore", [
        ("Durban",-29.87,31.03,20,0.10),("Indian Ocean South",-20.00,55.00,4200,0.22),
        ("Indian Ocean",-5.00,80.00,4000,0.18),("Malacca Strait",2.50,101.00,40,0.10),
        ("Singapore",1.29,103.85,25,0.08)]),
    "panama_losangeles": ("Panama \u2192 Los Angeles", [
        ("Panama Canal",9.08,-79.68,20,0.06),("Pacific Central",15.00,-95.00,3800,0.20),
        ("Baja California",25.00,-112.00,3000,0.16),("Los Angeles",33.74,-118.27,20,0.06)]),
    "tokyo_losangeles": ("Tokyo \u2192 Los Angeles", [
        ("Tokyo",35.65,139.84,20,0.07),("Pacific NW",40.00,160.00,5500,0.32),
        ("Mid-Pacific",42.00,-175.00,5800,0.35),("Pacific NE",38.00,-140.00,4500,0.28),
        ("Los Angeles",33.74,-118.27,20,0.06)]),
    "hamburg_newyork": ("Hamburg \u2192 New York", [
        ("Hamburg",53.55,9.99,15,0.04),("North Sea",55.00,3.00,50,0.10),
        ("North Atlantic",52.00,-30.00,4000,0.35),("New York",40.60,-74.05,20,0.06)]),
    "melbourne_shanghai": ("Melbourne \u2192 Shanghai", [
        ("Melbourne",-37.84,144.92,20,0.12),("Coral Sea",-18.00,155.00,3500,0.20),
        ("Philippine Sea",10.00,130.00,5000,0.22),("East China Sea",28.00,125.00,150,0.12),
        ("Shanghai",31.23,121.47,15,0.06)]),
    "jeddah_rotterdam": ("Jeddah \u2192 Rotterdam", [
        ("Jeddah",21.49,39.19,30,0.06),("Red Sea",25.00,36.00,1800,0.12),
        ("Suez",29.97,32.55,20,0.05),("Mediterranean",36.00,15.00,2500,0.12),
        ("Gibraltar",36.14,-5.35,300,0.08),("Rotterdam",51.95,4.14,25,0.04)]),
    "busan_losangeles": ("Busan \u2192 Los Angeles", [
        ("Busan",35.10,129.04,20,0.07),("Pacific NW",42.00,160.00,5500,0.32),
        ("Mid-Pacific",44.00,-175.00,5800,0.35),("Los Angeles",33.74,-118.27,20,0.06)]),
    "gibraltar_piraeus": ("Gibraltar \u2192 Piraeus", [
        ("Gibraltar",36.14,-5.35,300,0.08),("Alboran Sea",36.50,-2.00,1500,0.10),
        ("Sardinia",38.50,8.00,2800,0.14),("Ionian Sea",37.50,18.00,3000,0.13),
        ("Piraeus",37.94,23.65,40,0.06)]),
    "hongkong_singapore": ("Hong Kong \u2192 Singapore", [
        ("Hong Kong",22.30,114.17,25,0.08),("South China Sea",15.00,113.00,4000,0.15),
        ("Natuna Sea",4.00,108.00,80,0.10),("Singapore",1.29,103.85,25,0.08)]),
    "newyork_santos": ("New York \u2192 Santos", [
        ("New York",40.60,-74.05,20,0.06),("Caribbean",20.00,-65.00,4000,0.18),
        ("Equator Atlantic",0.00,-40.00,4200,0.16),("Santos",-23.98,-46.30,15,0.07)]),
    "capetown_singapore": ("Cape Town \u2192 Singapore", [
        ("Cape Town",-33.91,18.42,30,0.12),("Indian Ocean SW",-25.00,50.00,4500,0.24),
        ("Indian Ocean",-5.00,80.00,4000,0.18),("Singapore",1.29,103.85,25,0.08)]),
    "yokohama_singapore": ("Yokohama \u2192 Singapore", [
        ("Yokohama",35.44,139.64,20,0.07),("Philippine Sea",20.00,130.00,5000,0.22),
        ("South China Sea",10.00,115.00,4000,0.15),("Singapore",1.29,103.85,25,0.08)]),
    "antwerp_newyork": ("Antwerp \u2192 New York", [
        ("Antwerp",51.24,4.42,20,0.04),("English Channel",50.00,-2.00,50,0.10),
        ("North Atlantic",47.00,-35.00,4200,0.35),("New York",40.60,-74.05,20,0.06)]),
    "valencia_suez": ("Valencia \u2192 Suez", [
        ("Valencia",39.44,-0.32,25,0.06),("Mediterranean W",38.00,5.00,2500,0.12),
        ("Mediterranean E",35.00,20.00,3000,0.13),("Suez",29.97,32.55,20,0.05)]),
    "qingdao_rotterdam": ("Qingdao \u2192 Rotterdam", [
        ("Qingdao",36.07,120.38,20,0.06),("East China Sea",30.00,124.00,120,0.10),
        ("Malacca Strait",2.50,101.00,40,0.10),("Suez",29.97,32.55,20,0.05),
        ("Rotterdam",51.95,4.14,25,0.04)]),
    "colombo_singapore": ("Colombo \u2192 Singapore", [
        ("Colombo",6.94,79.84,20,0.08),("Bay of Bengal",8.00,88.00,3500,0.16),
        ("Malacca Strait",2.50,101.00,40,0.10),("Singapore",1.29,103.85,25,0.08)]),
    "houston_rotterdam": ("Houston \u2192 Rotterdam", [
        ("Houston",29.73,-94.98,15,0.06),("Gulf of Mexico",26.00,-88.00,2000,0.14),
        ("North Atlantic",40.00,-45.00,4500,0.30),("Rotterdam",51.95,4.14,25,0.04)]),
    "seattle_tokyo": ("Seattle \u2192 Tokyo", [
        ("Seattle",47.60,-122.33,20,0.07),("Pacific NE",48.00,-150.00,4500,0.28),
        ("Pacific NW",45.00,175.00,5500,0.32),("Tokyo",35.65,139.84,20,0.07)]),
    "piraeus_alexandria": ("Piraeus \u2192 Alexandria", [
        ("Piraeus",37.94,23.65,40,0.06),("Aegean Sea",36.00,26.00,1000,0.10),
        ("East Med",33.00,28.00,2500,0.12),("Alexandria",31.20,29.92,20,0.05)]),
    "vancouver_shanghai": ("Vancouver \u2192 Shanghai", [
        ("Vancouver",49.29,-123.11,25,0.07),("Pacific NE",50.00,-155.00,4500,0.28),
        ("Pacific NW",45.00,170.00,5500,0.32),("Shanghai",31.23,121.47,15,0.06)]),
    "istanbul_piraeus": ("Istanbul \u2192 Piraeus", [
        ("Istanbul Strait",41.10,29.05,35,0.05),("Dardanelles",40.20,26.40,60,0.07),
        ("Aegean Sea",39.00,25.00,1000,0.10),("Piraeus",37.94,23.65,40,0.06)]),
    "kobe_singapore": ("Kobe \u2192 Singapore", [
        ("Kobe",34.68,135.20,20,0.07),("East China Sea",28.00,125.00,150,0.12),
        ("South China Sea",12.00,115.00,4000,0.15),("Singapore",1.29,103.85,25,0.08)]),
    "algeciras_newyork": ("Algeciras \u2192 New York", [
        ("Algeciras",36.13,-5.45,300,0.08),("Atlantic Mid",38.00,-25.00,4000,0.30),
        ("Grand Banks",42.00,-48.00,150,0.25),("New York",40.60,-74.05,20,0.06)]),
    "dalian_singapore": ("Dalian \u2192 Singapore", [
        ("Dalian",38.92,121.63,20,0.06),("Yellow Sea",34.00,123.00,80,0.10),
        ("East China Sea",28.00,124.00,120,0.11),("South China Sea",12.00,114.00,4000,0.15),
        ("Singapore",1.29,103.85,25,0.08)]),
    "felixstowe_singapore": ("Felixstowe \u2192 Singapore", [
        ("Felixstowe",51.96,1.31,20,0.05),("Gibraltar",36.14,-5.35,300,0.08),
        ("Suez",29.97,32.55,20,0.05),("Indian Ocean",8.00,70.00,3800,0.18),
        ("Singapore",1.29,103.85,25,0.08)]),
    "longbeach_yokohama": ("Long Beach \u2192 Yokohama", [
        ("Long Beach",33.75,-118.19,20,0.06),("Pacific E",38.00,-140.00,4500,0.28),
        ("Mid-Pacific",42.00,-175.00,5800,0.35),("Yokohama",35.44,139.64,20,0.07)]),
    "genoa_alexandria": ("Genoa \u2192 Alexandria", [
        ("Genoa",44.41,8.93,25,0.06),("Tyrrhenian Sea",40.00,12.00,3000,0.13),
        ("Ionian Sea",36.00,18.00,3500,0.14),("Alexandria",31.20,29.92,20,0.05)]),
}

# ROUTES sozlugune ekle
route_entries = ""
for key,(name,wps) in new_routes.items():
    route_entries += '    "'+key+'": {\n        "name": "'+name+'",\n        "waypoints": [\n'
    for w in wps:
        route_entries += '            {"name":"'+w[0]+'","lat":'+str(w[1])+',"lon":'+str(w[2])+',"depth":'+str(w[3])+',"ssh":'+str(w[4])+'},\n'
    route_entries += '        ]\n    },\n'

# Son rotadan sonra ekle (atlantik'in kapanisindan sonra)
marker = '''            {"name":"N.Atlantic","lat":52.00,"lon":-30.00,"depth":3800,"ssh":0.38},
        ]
    },
}'''
replacement = '''            {"name":"N.Atlantic","lat":52.00,"lon":-30.00,"depth":3800,"ssh":0.38},
        ]
    },
''' + route_entries + '}'
content = content.replace(marker, replacement)

# Dropdown'a ekle
dropdown = ""
for key,(name,wps) in new_routes.items():
    dropdown += '        <option value="'+key+'">'+name+'</option>\n'

dmarker = '        <option value="atlantik">North Atlantic \u2014 Storm</option>'
content = content.replace(dmarker, dmarker+'\n'+dropdown.rstrip())

with open('app2.py','w',encoding='utf-8') as f:
    f.write(content)
print(str(len(new_routes))+' new routes added! Total: '+str(len(new_routes)+6))