# -*- coding: utf-8 -*-
with open('app2.py', encoding='utf-8') as f:
    content = f.read()

# initMap icine canli gemi katmani ekle
old = '''  }).addTo(map);
}'''

new = '''  }).addTo(map);
  startAIS();
}

var aisMarkers={};
var aisWS=null;
function startAIS(){
  if(aisWS) return;
  var API_KEY="d451be71cb01176584568bcd91c632349587eaff";
  aisWS=new WebSocket("wss://stream.aisstream.io/v0/stream");
  aisWS.onopen=function(){
    aisWS.send(JSON.stringify({
      APIKey:API_KEY,
      BoundingBoxes:[[[-90,-180],[90,180]]],
      FilterMessageTypes:["PositionReport"]
    }));
  };
  aisWS.onmessage=function(ev){
    try{
      var msg=JSON.parse(ev.data);
      if(msg.MessageType!=="PositionReport") return;
      var r=msg.Message.PositionReport;
      var mmsi=r.UserID;
      var lat=r.Latitude, lon=r.Longitude;
      var cog=r.Cog||0;
      if(aisMarkers[mmsi]){
        aisMarkers[mmsi].setLatLng([lat,lon]);
      }else{
        if(Object.keys(aisMarkers).length>500) return;
        var icon=L.divIcon({
          className:"ais-ship",
          html:"<div style='width:8px;height:8px;background:#00E5B0;border-radius:50%;box-shadow:0 0 6px #00E5B0;transform:rotate("+cog+"deg)'></div>",
          iconSize:[8,8]
        });
        var m=L.marker([lat,lon],{icon:icon}).addTo(map);
        var name=(msg.MetaData&&msg.MetaData.ShipName)?msg.MetaData.ShipName.trim():"Vessel";
        m.bindTooltip("<b>"+name+"</b><br>MMSI: "+mmsi+"<br>Speed: "+(r.Sog||0)+" kn",{className:"leaflet-tooltip-dark"});
        aisMarkers[mmsi]=m;
      }
    }catch(e){}
  };
  aisWS.onclose=function(){aisWS=null;setTimeout(startAIS,5000);};
}'''

content = content.replace(old, new, 1)

with open('app2.py','w',encoding='utf-8') as f:
    f.write(content)
print("Live AIS ship tracking added!")