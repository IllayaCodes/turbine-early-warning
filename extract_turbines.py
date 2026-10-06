import shutil
import sys
import zipfile
from pathlib import Path

zpath = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zpath) as z:
    for info in z.infolist():
        base = Path(info.filename).name
        if base.startswith("Turbine_Data_Kelmarsh_") and not (out / base).exists():
            with z.open(info) as src, open(out / base, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print("extracted", base)