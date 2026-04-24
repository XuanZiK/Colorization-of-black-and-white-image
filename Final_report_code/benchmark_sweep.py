import matplotlib.pyplot as plt

from benchmark_two_models import benchmark_models

SAMPLE_STEPS = list(range(100, 1001, 100))  # 100,200,...,1000


def run_sweep():
    records = []
    for total in SAMPLE_STEPS:
        print(f"\n=== Running benchmark for total_samples={total} ===")
        results = benchmark_models(total_samples=total)
        for name, processed, elapsed, avg, save_dir in results:
            records.append({
                "model": name,
                "samples": processed,
                "elapsed": elapsed,
                "avg": avg,
            })
    return records


def plot_results(records):
    models = sorted(set(r["model"] for r in records))
    plt.figure(figsize=(8, 5))
    for m in models:
        xs = [r["samples"] for r in records if r["model"] == m]
        ys = [r["avg"] for r in records if r["model"] == m]
        plt.plot(xs, ys, marker="o", label=m)
    plt.xlabel("Total samples")
    plt.ylabel("Avg time per image (s)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("benchmark_sweep.png")
    print("Saved plot to benchmark_sweep.png")


def main():
    records = run_sweep()
    plot_results(records)


if __name__ == "__main__":
    main()
