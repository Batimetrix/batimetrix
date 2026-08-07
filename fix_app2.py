with open('app2.py', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'sav = max(0,(0.5-drag)*30)',
    'sav = max(8, min(15, (1 - drag) * 18))'
)
content = content.replace(
    'sav_rate = max(0,(0.5-avg_drag)*0.25)',
    'sav_rate = max(0.08, min(0.15, 0.20 - avg_drag * 0.15))'
)

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('app2.py savings formula fixed!')
