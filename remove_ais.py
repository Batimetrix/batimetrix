with open('app2.py', encoding='utf-8') as f:
    content = f.read()

content = content.replace('  try{startAIS();}catch(e){console.log(e);}', '')

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Removed startAIS call')
