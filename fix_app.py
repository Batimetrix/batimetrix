import re

with open('app.py', encoding='utf-8') as f:
    content = f.read()

old = 'model = GucluPINN()\nmodel.load_state_dict(torch.load("batimetrix_model_v2.pt", weights_only=True))\nmodel.eval()\nprint("Model loaded!")'

new = '''model = None
try:
    model = GucluPINN()
    model.load_state_dict(torch.load("batimetrix_model_v2.pt", weights_only=True))
    model.eval()
    print("Model loaded!")
except Exception as e:
    print(f"Model not found, using physics formula: {e}")
    model = None'''

content = content.replace(old, new)

old2 = 'def tahmin_drag(lat, lon, depth, ssh, swh, speed, draft):\n    inp = torch.tensor([[\n        (lat+70)/150, (lon+180)/360, depth/6000,\n        (ssh+2)/4, swh/20, speed/25, draft/22\n    ]]).float()\n    with torch.no_grad():\n        return model(inp).item()'

new2 = '''def tahmin_drag(lat, lon, depth, ssh, swh, speed, draft):
    if model is not None:
        inp = torch.tensor([[
            (lat+70)/150, (lon+180)/360, depth/6000,
            (ssh+2)/4, swh/20, speed/25, draft/22
        ]]).float()
        with torch.no_grad():
            return model(inp).item()
    else:
        re = (speed * 0.5144) * draft / 1.05e-6
        cf = 0.075 / ((np.log10(re) - 2) ** 2) if re > 0 else 0.002
        wave_factor = 1 + 0.015 * swh ** 1.5
        depth_factor = 1 + 0.1 * max(0, 1 - depth/50)
        drag = cf * wave_factor * depth_factor * (speed/10) ** 1.8
        return min(max(drag * 15, 0.05), 0.95)'''

content = content.replace(old2, new2)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done!')
