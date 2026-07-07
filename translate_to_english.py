import os
import re

# Turkce -> English sozluk
SOZLUK = {
    # Degisken isimleri
    "gemi_tipi": "vessel_type",
    "gemi_tipleri": "vessel_types",
    "gemi_adi": "vessel_name",
    "gemi_profil": "vessel_profile",
    "gemi_filosu": "vessel_fleet",
    "guzergah": "route",
    "guzergahlar": "routes",
    "noktalar": "waypoints",
    "sonuc_noktalar": "result_waypoints",
    "tasarruf": "savings",
    "tasarruf_oran": "savings_rate",
    "tasarruf_usd": "savings_usd",
    "tasarruf_pct": "savings_pct",
    "tasarruf_oran": "savings_rate",
    "para_tasarruf": "cost_savings",
    "yakit": "fuel",
    "yakit_gun": "fuel_per_day",
    "yakit_fiyat": "fuel_price",
    "yakit_turu": "fuel_type",
    "yillik_yakit_usd": "annual_fuel_usd",
    "derinlik": "depth",
    "hiz": "speed",
    "tastak": "draft",
    "sefer_gun": "voyage_days",
    "sefer_mesafe": "voyage_distance",
    "ort_drag": "avg_drag",
    "drag_toplam": "drag_total",
    "drag_base": "drag_base",
    "sonuclar": "results",
    "profil": "profile",
    "mevsim": "season",
    "mevsimler": "seasons",
    "iyilesen": "improved",
    "not_iyilestiren": "rating_improved",
    "guzergah_tasarruf": "route_savings",
    "toplam_tasarruf_usd": "total_savings_usd",
    "toplam_guzergah": "total_routes",
    "koordinatlar": "coordinates",
    "isim": "name",
    "emoji": "emoji",
    "aciklama": "description",
    "renk": "color",
    "rota": "route",
    "rota_renk": "route_color",
    "ort_kazan": "avg_earnings",
    "ort_drag": "avg_drag",
    "ort_tasarruf": "avg_savings",
    "senaryo": "scenario",
    "senaryolar": "scenarios",
    "gercek_gunluk_ton": "real_daily_tons",
    "gunluk_yakit_ton": "daily_fuel_tons",
    "yillik_yakit_ton": "annual_fuel_tons",
    "tasarruf_ton": "savings_tons",
    "co2_azalma": "co2_reduction",
    "co2_b_g": "co2_baseline_g",
    "co2_a_g": "co2_optimized_g",
    "req_cii": "required_cii",
    "attained": "attained",
    "gercek_cii": "actual_cii",
    "referans_cii": "reference_cii",
    "not_base": "rating_base",
    "not_opt": "rating_opt",
    "degisti": "changed",
    "ab": "label_base",
    "ao": "label_opt",
    "ob": "ratio_base",
    "oo": "ratio_opt",
    "azalma": "reduction",
    "yogunluk": "intensity",
    "yagis_yogunluk": "precipitation_intensity",
    "swh_etkisi": "swh_effect",
    "hava": "weather",
    "durum": "status",
    "nokta_renk": "waypoint_color",
    "p_cii": "cii_params",
    "p": "params",
    "tas": "sav",
    "tas_oran": "savings_rate",
    "oran": "ratio",
    "sinir": "limit",
    "sinirlar": "limits",
    "katmanlar": "layers",
    "giris": "input_layer",
    "cikis": "output_layer",
    "inp": "inp",

    # Print mesajlari
    "Model yuklendi!": "Model loaded!",
    "Model yuklendi": "Model loaded",
    "tamamlandi": "completed",
    "tamamlandi!": "completed!",
    "bulundu": "found",
    "kaydedildi": "saved",
    "yuklendi": "loaded",
    "Harita kaydedildi": "Map saved",
    "Grafik kaydedildi": "Chart saved",
    "Gemi tipi analizi tamamlandi": "Vessel type analysis completed",
    "Gercekci analiz tamamlandi": "Realistic analysis completed",
    "CII modulu tamamlandi": "CII module completed",
    "Gorsellestirme tamamlandi": "Visualization completed",

    # Yorum satirlari
    "# Guzergah noktalari": "# Route waypoints",
    "# Her nokta icin drag hesapla": "# Calculate drag for each waypoint",
    "# Yillik hesaplar": "# Annual calculations",
    "# Gemi Profilleri": "# Vessel profiles",
    "# Gemi tipi": "# Vessel type",
    "# Test Senaryolari": "# Test scenarios",
    "# Sonuclar": "# Results",
    "# Ozet": "# Summary",
    "# Model": "# Model",
    "# Harita": "# Map",
    "# Bilgi Paneli": "# Info panel",
    "# Guzergah cizgisi": "# Route line",
    "# Nokta rengi": "# Waypoint color",
    "# Genel Ozet": "# General summary",
    "# Gercekcilik Notu": "# Realism note",
    "# Pitch Ozeti": "# Pitch summary",
    "# Filo": "# Fleet",
    "# Detay": "# Detail",
    "# CII hesapla": "# Calculate CII",
    "# CO2 hesapla": "# Calculate CO2",
    "# Vizkozite": "# Viscosity",
    "# Guzergah analizi": "# Route analysis",
    "# SWOT Verisi Cek": "# Fetch SWOT data",
    "# GPM Verisi Cek": "# Fetch GPM data",
    "# Veri Ozeti": "# Data summary",
    "# 3 Uydu Ozet": "# 3-satellite summary",
    "# Birlestirilen Veri": "# Combined data",
    "# Fizik Kayip Fonksiyonlari": "# Physics loss functions",
    "# Optimizer": "# Optimizer",
    "# Guzergah": "# Route",
    "# IMO CII": "# IMO CII",
    "# Not sinir": "# Rating boundaries",
    "# Azaltma faktoru": "# Reduction factor",
}

def dosyayi_cevir(dosya_yolu):
    with open(dosya_yolu, 'r', encoding='utf-8') as f:
        icerik = f.read()

    orijinal = icerik

    # Sozlukten kelime kelime degistir
    for turkce, ingilizce in SOZLUK.items():
        # Tam kelime eslesimi (degisken isimleri icin)
        icerik = re.sub(
            r'\b' + re.escape(turkce) + r'\b',
            ingilizce,
            icerik
        )

    if icerik != orijinal:
        with open(dosya_yolu, 'w', encoding='utf-8') as f:
            f.write(icerik)
        return True
    return False

# Tum .py dosyalarini isle
print("=== BATIMETRIX - English Translation Script ===\n")
klasor = "."
degistirilen = 0
atlanan = 0

for dosya in os.listdir(klasor):
    if dosya.endswith(".py") and dosya != "translate_to_english.py":
        yol = os.path.join(klasor, dosya)
        if dosyayi_cevir(yol):
            print(f"  [TRANSLATED] {dosya}")
            degistirilen += 1
        else:
            print(f"  [SKIPPED]    {dosya} (no changes needed)")
            atlanan += 1

print(f"\nDone! {degistirilen} files translated, {atlanan} skipped.")
print("\nNext step: git add . && git commit -m 'Codebase translated to English'")