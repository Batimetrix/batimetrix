with open('app.py', encoding='utf-8') as f:
    content = f.read()

# Eski tasarruf formulu
old = 'savings_rate     = max(0, (0.5 - avg_drag) * 0.25)'
new = 'savings_rate     = max(0.08, min(0.15, 0.20 - avg_drag * 0.15))'
content = content.replace(old, new)

# Nokta bazi tasarruf
old2 = 'sav  = max(0, (0.5 - drag) * 30)'
new2 = 'sav  = max(8, min(15, (1 - drag) * 18))'
content = content.replace(old2, new2)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Savings formula fixed!')
