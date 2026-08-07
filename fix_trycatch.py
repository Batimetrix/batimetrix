with open('app2.py', encoding='utf-8') as f:
    content = f.read()

content = content.replace('  startAIS();', '  try{startAIS();}catch(e){console.log(e);}')

with open('app2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Wrapped startAIS in try-catch')
