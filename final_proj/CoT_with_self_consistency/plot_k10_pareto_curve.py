import json
import matplotlib.pyplot as plt
import numpy as np

# Apply clean academic plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_k10_pareto_curves(json_path="self_consistency_150_results.json"):
    # 1. Load the evaluation results JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    k10_summary = data["k10_summary_metrics"]
    per_subset = k10_summary["per_subset_results"]

    # 2. Extract Data Points for k=10
    # Individual Subsets
    subsets = list(per_subset.keys())
    qualities = [per_subset[s]["mean_quality"] for s in subsets]
    tokens = [per_subset[s]["mean_tokens"] for s in subsets]
    calls = [per_subset[s]["mean_calls"] for s in subsets]

    # Aggregate / Pooled Points
    overall_quality = k10_summary["pooled_quality_all_5_subsets_macro"]
    overall_tokens = k10_summary["overall_mean_tokens_per_problem"]
    overall_calls = k10_summary["overall_mean_calls_per_problem"]

    group_a_quality = k10_summary["pooled_quality_group_A_macro"]
    group_a_tokens = k10_summary["group_A_mean_tokens"]
    group_a_calls = k10_summary["group_A_mean_calls"]

    group_b_quality = k10_summary["pooled_quality_group_B_macro"]
    group_b_tokens = k10_summary["group_B_mean_tokens"]
    group_b_calls = k10_summary["group_B_mean_calls"]

    # 3. Create Figure with Two Subplots (Tokens vs Calls)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # --- Subplot 1: Quality vs. Mean Tokens per Problem ---
    # Plot individual subsets
    ax1.scatter(tokens, qualities, color="#2b5c8f", s=100, zorder=3, label="Individual Subsets")
    for i, txt in enumerate(subsets):
        ax1.annotate(txt, (tokens[i], qualities[i]), textcoords="offset points", 
                     xytext=(0, 8), ha='center', fontsize=9, fontweight='bold', color="#2b5c8f")

    # Plot Group A, Group B, and Overall Macro Points
    ax1.scatter([group_a_tokens], [group_a_quality], color="#d95f02", s=140, marker="s", zorder=4, label="Group A Macro (Short/Exact)")
    ax1.annotate("Group A Macro", (group_a_tokens, group_a_quality), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=9, fontweight='bold', color="#d95f02")

    ax1.scatter([group_b_tokens], [group_b_quality], color="#7570b3", s=140, marker="^", zorder=4, label="Group B Macro (ASQA Long)")
    ax1.annotate("Group B Macro", (group_b_tokens, group_b_quality), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold', color="#7570b3")

    ax1.scatter([overall_tokens], [overall_quality], color="#1b9e77", s=180, marker="*", zorder=5, label="Overall Macro (All 5 Subsets)")
    ax1.annotate("Overall Macro", (overall_tokens, overall_quality), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=10, fontweight='bold', color="#1b9e77")

    ax1.set_title("k=10 Pareto Frontier: Quality vs. Mean Tokens", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Mean Tokens per Problem (Input + Output)", fontsize=11)
    ax1.set_ylabel("Macro-Average Quality Score", fontsize=11)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # --- Subplot 2: Quality vs. Mean LLM API Calls ---
    # Plot individual subsets
    ax2.scatter(calls, qualities, color="#2b5c8f", s=100, zorder=3, label="Individual Subsets")
    for i, txt in enumerate(subsets):
        ax2.annotate(txt, (calls[i], qualities[i]), textcoords="offset points", 
                     xytext=(0, 8), ha='center', fontsize=9, fontweight='bold', color="#2b5c8f")

    # Plot Group A, Group B, and Overall Macro Points
    ax2.scatter([group_a_calls], [group_a_quality], color="#d95f02", s=140, marker="s", zorder=4, label="Group A Macro (Short/Exact)")
    ax2.annotate("Group A Macro", (group_a_calls, group_a_quality), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=9, fontweight='bold', color="#d95f02")

    ax2.scatter([group_b_calls], [group_b_quality], color="#7570b3", s=140, marker="^", zorder=4, label="Group B Macro (ASQA Long)")
    ax2.annotate("Group B Macro", (group_b_calls, group_b_quality), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold', color="#7570b3")

    ax2.scatter([overall_calls], [overall_quality], color="#1b9e77", s=180, marker="*", zorder=5, label="Overall Macro (All 5 Subsets)")
    ax2.annotate("Overall Macro", (overall_calls, overall_quality), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=10, fontweight='bold', color="#1b9e77")

    ax2.set_title("k=10 Pareto Frontier: Quality vs. Mean API Calls", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Mean LLM API Calls per Problem (k x CoT steps + retries)", fontsize=11)
    ax2.set_ylabel("Macro-Average Quality Score", fontsize=11)
    ax2.legend(loc="lower right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Global layout adjustments
    fig.suptitle(f"Self-Consistency Performance & Cost Analysis (Config: k=10 | Local Free Ollama Cost: $0.00)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save high-resolution figure
    output_filename = "k10_pareto_curve.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Successfully generated and saved k=10 Pareto plot to '{output_filename}'!")
    
    plt.show()

if __name__ == "__main__":
    plot_k10_pareto_curves()