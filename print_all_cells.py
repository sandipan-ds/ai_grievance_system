import json

path = r"c:\Users\sandi\Desktop\ML Working Folder\ai_grievance_system\notebook\ai_grievance_system.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get("cells", [])):
    print(f"=== Cell Index {i} (1-indexed {i+1}) [{cell['cell_type']}] ===")
    source = cell.get("source", [])
    if source:
        print("Line 1:", repr(source[0]))
    else:
        print("Empty cell")
    print()
