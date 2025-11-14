import time, csv, pathlib
import requests
import pandas as pd
import matplotlib.pyplot as plt

BASE_URL = "http://serve-sentiment-test-env.eba-bcwv7z2e.us-east-1.elasticbeanstalk.com/predict"
N_CALLS = 100
TIMEOUT = 15
OUT_DIR = pathlib.Path("results"); OUT_DIR.mkdir(exist_ok=True)

TESTS = {
    "fake_1": "The weather is controlled by secret government machines hidden in the mountains.",
    "fake_2": "The world has been taken over by a happiness virus that makes everyone smile all the time.",
    "real_1": "The Blue Jays lose in Game 7 of the World Series to the Los Angeles Dodgers.",
    "real_2": "The government shutdown has been ended after long negotiations in the Senate.",
}

FIELDS = ["case","iter","status_code","latency_ms","prediction","timestamp_start_iso"]

def run_case(name, text):
    rows = []
    for i in range(1, N_CALLS + 1):
        t0 = time.time()
        try:
            r = requests.post(BASE_URL, json={"text": text}, timeout=TIMEOUT)
            latency = (time.time() - t0) * 1000
            pred = None
            try:
                pred = r.json().get("prediction")
            except Exception:
                pred = None
            rows.append({
                "case": name,
                "iter": i,
                "status_code": r.status_code,
                "latency_ms": round(latency, 2),
                "prediction": pred,
                "timestamp_start_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
            })
        except requests.exceptions.RequestException:
            latency = (time.time() - t0) * 1000
            rows.append({
                "case": name,
                "iter": i,
                "status_code": -1,
                "latency_ms": round(latency, 2),
                "prediction": None,
                "timestamp_start_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
            })

    with open(OUT_DIR / f"{name}_latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    return rows

def main():
    all_rows = []
    for name, text in TESTS.items():
        print(f"Running {name}…")
        all_rows.extend(run_case(name, text))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "latency_all.csv", index=False)

    ok = df[df["status_code"] == 200]
    plt.figure(figsize=(9,6))

    ax = ok.boxplot(column="latency_ms", by="case", grid=False, return_type=None)
    plt.title("API Latency per Test Case (ms)"); plt.suptitle("")
    plt.xlabel("Test Case"); plt.ylabel("Latency (ms)")

    # Compute averages
    avgs = ok.groupby("case")["latency_ms"].mean().round(2)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "latency_boxplot.png", dpi=300)

    with open(OUT_DIR / "latency_averages.txt", "w") as f:
        f.write("Average latency (ms) by case\n")
        for k, v in avgs.items():
            f.write(f"{k}: {v}\n")

    print("\nAverages (ms):\n", avgs.to_string())

if __name__ == "__main__":
    main()