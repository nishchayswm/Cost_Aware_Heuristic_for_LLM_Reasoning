"""Report one selected experiment; never pool old and repaired implementations."""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_all_results(input_files):
    rows = []
    for filename in input_files:
        with open(filename, encoding='utf-8') as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError('No results in the selected files')
    versions = {row.get('engine_version', 'legacy') for row in rows}
    if len(versions) != 1:
        raise ValueError('Do not mix legacy and repaired engine results')
    if len({row.get('experiment_id', 'unspecified') for row in rows}) != 1:
        raise ValueError('Select a single experiment; do not pool different runs')
    tos_versions = {row.get('baseline_version', 'unknown') for row in rows
                    if str(row['method']).startswith('Thought of Search')}
    if len(tos_versions) > 1:
        raise ValueError('Do not mix different Thought of Search adaptation versions')
    df = pd.DataFrame(rows)
    if 'seed' not in df:
        df['seed'] = 'legacy'
    if 'dataset' not in df:
        df['dataset'] = 'custom'
    if df.duplicated(['dataset', 'method', 'seed', 'problem_id']).any():
        raise ValueError('Duplicate problem/method/seed records; select a single experiment')
    df['outcome'] = ['missing' if pd.isna(row.get('ans', row.get('final_answer'))) else
                     'correct' if row.get('correct') is True else 'wrong' for row in rows]
    df['correct'] = df['outcome'].eq('correct')
    return df


def generate_report_and_plots(output_dir='kaggle/results', input_files=None):
    if not input_files:
        raise ValueError('Select input_files explicitly so historical experiments are not pooled')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = load_all_results(input_files)
    df['missing'] = df['outcome'].eq('missing')
    df['wrong'] = df['outcome'].eq('wrong')
    summary = df.groupby(['dataset', 'method']).agg(
        Runs=('correct', 'size'), Correct=('correct', 'sum'),
        Missing=('missing', 'sum'), Wrong=('wrong', 'sum'),
        Accuracy=('correct', lambda x: x.mean()*100),
        Mean_Calls=('calls', 'mean'), Mean_Tokens=('tokens', 'mean'),
        Avg_Time_Sec=('sec', 'mean')).reset_index()
    per_seed = df.groupby(['dataset', 'method', 'seed']).agg(
        Accuracy=('correct', lambda x: x.mean()*100), Mean_Calls=('calls', 'mean'),
        Mean_Tokens=('tokens', 'mean')).reset_index()
    spread = per_seed.groupby(['dataset', 'method']).agg(
        Seeds=('seed', 'size'), Accuracy_Seed_SD=('Accuracy', 'std')).reset_index()
    summary = summary.merge(spread, on=['dataset', 'method']).sort_values(['dataset', 'Accuracy'],
                                                                           ascending=[True, False])
    references = summary.loc[summary.method.eq('Tree-of-Thoughts (LLM-Judge)')].set_index('dataset').Mean_Calls
    if not references.empty:
        denominators = summary.dataset.map(references).replace(0, float('nan'))
        summary['Call_Savings_vs_ToT_%'] = (1-summary.Mean_Calls/denominators)*100
    summary.to_csv(output / 'summary_table.csv', index=False)
    per_seed.to_csv(output / 'per_seed_summary.csv', index=False)
    # Save failure counts and terminal statuses alongside accuracy.
    if 'status' in df:
        df.groupby(['dataset', 'method', 'status']).size().rename('count').to_csv(output / 'termination_counts.csv')
    print(summary.to_string(index=False))
    datasets = list(summary.dataset.unique())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 7*len(datasets)), squeeze=False)
    for dataset, ax in zip(datasets, axes[:, 0]):
        for _, row in summary.loc[summary.dataset.eq(dataset)].iterrows():
            ax.scatter(row.Mean_Calls, row.Accuracy, s=90)
            ax.annotate(row.method, (row.Mean_Calls, row.Accuracy), xytext=(6, 6),
                        textcoords='offset points', fontsize=8)
        ax.set(xlabel='Mean LLM calls (all retries and reviews included)', ylabel='Accuracy (%)',
               ylim=(-5, 105), title=dataset + ': accuracy and inference cost')
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / 'pareto_frontier.png', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 5*len(datasets)), squeeze=False)
    for dataset, ax in zip(datasets, axes[:, 0]):
        subset = summary.loc[summary.dataset.eq(dataset)]
        ax.barh(subset.method, subset.Mean_Tokens)
        ax.set(xlabel='Mean prompt + completion tokens', title=dataset + ': token consumption')
    fig.tight_layout()
    fig.savefig(output / 'token_consumption.png', dpi=200)
    plt.close(fig)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--output', default='kaggle/results/report')
    args = parser.parse_args()
    generate_report_and_plots(args.output, args.inputs)
