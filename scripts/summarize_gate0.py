import json
from pathlib import Path

payload = json.loads(Path("reports/gate0_audit_raw.json").read_text(encoding="utf-8"))
young = payload["young"]
wsc = payload["wsc"]
print("EXP", young["experimental"]["treatment_true"], young["experimental"]["treatment_false"], young["experimental"]["treatment_rate"])
print("OBS", young["observed"]["treatment_true"], young["observed"]["treatment_false"], young["observed"]["treatment_rate"])
print("DROP", young["experimental"]["dropped_gt_44"], young["observed"]["dropped_gt_44"], young["observed"]["dropped_treated_gt_44"], young["observed"]["dropped_control_gt_44"])
print("MISS", young["experimental"]["n_missing_cells"], young["observed"]["n_missing_cells"])
print("EXP_Y", young["experimental"]["outcomes"])
print("OBS_Y", young["observed"]["outcomes"])
for item in young["files"]:
    print("FILE", item["name"], item["bytes"], item["sha256"])
print("WSC_RDA", wsc["github_rda"])
print("CRAN_VER", wsc["cran"]["cran_version_line"], wsc["cran"]["cran_license_line"])
print("CRAN_TGZ", wsc["cran"]["bytes"], wsc["cran"]["sha256"])
print("CRAN_RDA", wsc["cran"]["cran_rda"])
print("INTEGRITY", wsc["coding_check_difference_in_means"])
print("DUP", wsc["duplicate_rows"], "QED", wsc["n_qed_concordant"], wsc["qed_treatment_equals_preference"])
print("PREF", wsc["n_prefer_math"], wsc["n_prefer_vocab"], wsc["n_math_assigned"], wsc["n_vocab_assigned"])
missing = [row for row in wsc["column_summaries"] if row["n_missing"]]
print("WSC_MISSING", missing)
for item in young["saved_outputs"]:
    content = item.get("content", {})
    print("PKL", item["name"], item["bytes"], item.get("loaded"), content.get("type"), content.get("keys") or content.get("columns"))
    if content.get("type") == "dict":
        print("  VALS", json.dumps(content.get("values"), default=str)[:1500])
    if content.get("preview"):
        print("  PREVIEW", json.dumps(content.get("preview"), default=str)[:1500])
