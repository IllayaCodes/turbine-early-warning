import sys
from pathlib import Path

path = Path(sys.argv[1])
with open(path, encoding="utf-8", errors="replace") as f:
    lines = [f.readline() for _ in range(15)]

print("--- first 15 lines (cut to 110 chars) ---")
for i, line in enumerate(lines):
    print(i, line[:110].rstrip())

for line in lines:
    if "Date and time" in line:
        cols = [c.strip() for c in line.lstrip("# ").split(",")]
        Path("docs/columns.txt").write_text("\n".join(cols), encoding="utf-8")
        print()
        print(len(cols), "columns, all saved to docs/columns.txt")
        print("--- columns mentioning temp / wind speed / power / speed ---")
        keys = ("temp", "wind speed", "power", "speed")
        for c in cols:
            if any(k in c.lower() for k in keys):
                print(" -", c)
        break
