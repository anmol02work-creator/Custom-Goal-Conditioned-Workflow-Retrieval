import json
import ast

nb = json.load(open('JARVIS_Model_A_Workflow_Retrieval.ipynb', encoding='utf-8'))
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # Filter out bash lines like !pip
        py_lines = [l for l in source.splitlines() if not l.strip().startswith('!')]
        py_code = "\n".join(py_lines)
        try:
            compile(py_code, f"cell_{idx}", "exec")
        except SyntaxError as e:
            print(f"Syntax error in cell {idx}: {e}")
            raise

print("All code cells successfully compiled without Python syntax errors!")
