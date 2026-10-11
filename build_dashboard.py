import json
import sys
from pathlib import Path

import yaml

from src.turbine_ew.dashboard import build_payload, write_dashboard
from src.turbine_ew.pipeline import daily_residuals

SECTION = "event"
MODEL = "linear"
OUT = Path("docs/dashboard.html")
DATA = Path("docs/dashboard-data.json")

if "reuse" in sys.argv and DATA.exists():
    # Page only: reuse the saved numbers, so it takes seconds instead of minutes.
    payload = json.loads(DATA.read_text(encoding="utf-8"))
else:
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    daily = daily_residuals(cfg, [MODEL], section=SECTION)[MODEL]
    payload = build_payload(cfg, daily, section=SECTION, model_name=MODEL)
    DATA.write_text(json.dumps(payload), encoding="utf-8")

write_dashboard(payload, OUT)

print("wrote", OUT)
print("days shown:", len(payload["dates"]), "| alerts:", len(payload["alerts"]))
for a in payload["alerts"]:
    print(" ", a["alertDate"], "turbine", a["turbine"], a["target"], "lead", a["leadDays"])
