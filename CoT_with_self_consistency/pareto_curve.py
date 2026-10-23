import json
import matplotlib.pyplot as plt
import numpy as np

# Set dark mode preference or clean academic style matching your preferences
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def load_evaluation_data(json_path="self_consistency_150_results.json"):
    """Loads the main evaluation summary and problem-level results."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def plot_pareto_curves():
    # Load data (assuming we have generated points for k=1, 3, 5, 8, 10)
    # If running a multi-k replay, you can loop over your multi-k summary dictionaries.
    data = load_evaluation_data()
    summary = data["summary_metrics"]
    per_subset = summary["per_subset_results"]

    # Extract metrics for plotting
    subsets = list(per_subset.keys())
    qualities = [per_subset[s]["mean_quality"] for s in subsets]
    tokens = [per_subset[s]["mean_tokens"] for s in subsets]
    calls = [per_subset[s]["mean_calls"] for s in subsets]

    # Create figure with 2 subplots (Pareto X-axes: Tokens vs Calls)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Quality vs. Mean Tokens per Problem ---
    ax1.scatter(tokens, qualities, color="#2b5c8f", s=100, zorder=3, label="Subsets")
    for i, txt in enumerate(subsets):
        ax1.annotate(txt, (tokens[i], qualities[i]), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')

    ax1.set_title("Pareto Frontier: Quality vs. Tokens", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Mean Tokens per Problem", fontsize=11)
    ax1.set_ylabel("Macro-Average Quality Score", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # --- Plot 2: Quality vs. Mean Calls per Problem ---
    ax2.scatter(calls, qualities, color="#d95f02", s=100, zorder=3, label="Subsets")
    for i, txt in enumerate(subsets):
        ax2.annotate(txt, (calls[i], qualities[i]), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')

    ax2.set_title("Pareto Frontier: Quality vs. API Calls ($k$)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Mean Calls ($k$) per Problem", fontsize=11)
    ax2.set_ylabel("Macro-Average Quality Score", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Global layout adjustments
    fig.suptitle(f"Self-Consistency Performance & Cost Trade-offs ({summary['config']})", fontsize=14, y=1.02)
    plt.tight_layout()

    # Save figure high-res for reports/presentations
    output_filename = "pareto_curves_output.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Successfully saved Pareto plots to '{output_filename}'!")
    
    plt.show()

if __name__ == "__main__":
    plot_pareto_curves()