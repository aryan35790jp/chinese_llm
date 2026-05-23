import json
nb = json.load(open(r'c:\chinese_llm_composition\notebooks\colab_expand.ipynb', encoding='utf8'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    print(f'--- CELL {i} ({c["cell_type"]}) ---')
    print(src[:1200])
    print()
