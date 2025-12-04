import json
import sys

REQUIRED_FIELDS = [
    "id",
    "lat",
    "lon",
    "has_solar",
    "confidence",
    "panel_count",
    "area_sqm",
    "capacity_kw",
    "qc_status",
    "reason_codes",
    "mask_path",
]

def validate(path):
    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("[ERROR] results.json must be a JSON array.")
        sys.exit(1)

    ok = True

    for i, item in enumerate(data):
        # Check required keys
        for field in REQUIRED_FIELDS:
            if field not in item:
                print(f"[ERROR] Row {i} missing field: {field}")
                ok = False

        # Basic type sanity checks
        try:
            _ = str(item["id"])
            float(item["lat"])
            float(item["lon"])
            int(item["has_solar"])
            float(item["confidence"])
            int(item["panel_count"])
            float(item["area_sqm"])
            float(item["capacity_kw"])
            assert item["qc_status"] in ["VERIFIABLE", "NOT_VERIFIABLE"]
            assert isinstance(item["reason_codes"], list)
        except Exception as e:
            print(f"[ERROR] Row {i} has invalid types: {e}")
            ok = False

    if ok:
        print("[✔] JSON validation passed. Format looks good.")
    else:
        print("[✖] JSON validation failed.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/validate_results_json.py <path_to_results.json>")
        sys.exit(1)

    validate(sys.argv[1])
