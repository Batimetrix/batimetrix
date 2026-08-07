import torch, re, shutil, glob

# app.py hangi .pt dosyasini yukluyor bul
with open('app.py', 'r', encoding='utf-8') as f:
    src = f.read()
m = re.search(r'torch\.load\(\s*["\']([^"\']+\.pt)["\']', src)
pt_file = m.group(1) if m else 'batimetrix_guclu.pt'
print(f"Model dosyasi: {pt_file}")

# Yedek al
shutil.copy(pt_file, pt_file + '.bak')
print(f"Yedek: {pt_file}.bak")

# Key'leri cevir
sd = torch.load(pt_file, weights_only=True)
yeni = {}
for k, v in sd.items():
    nk = (k.replace('giris.', 'input_layer.')
           .replace('katmanlar.', 'layers.')
           .replace('cikis.', 'output_layer.'))
    yeni[nk] = v
    if nk != k:
        print(f"  {k}  ->  {nk}")

torch.save(yeni, pt_file)
print(f"\nTAMAM! {len(yeni)} key cevrildi. Simdi: python app.py")
