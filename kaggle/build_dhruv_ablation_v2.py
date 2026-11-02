"""Build a separate ablation notebook with bounded Ollama transport recovery."""
from pathlib import Path

import build_dhruv_ablation_notebook as base


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'kaggle' / 'Dhruv_five_factor_ablations_recovery_v2.ipynb'


def build():
    base.OUTPUT = OUTPUT
    base.RUNNER = ROOT / 'kaggle' / 'dhruv_ablation_runner_v2.py'
    base.DIRECTORIES = base.DIRECTORIES.replace(
        'dhruv_ablation_proj', 'dhruv_ablation_recovery_v2')
    base.SETTINGS = base.SETTINGS.replace(
        'RETRY_FAILED = True',
        "RETRY_FAILED = True\nMIGRATION_SOURCE = ''  # Optional old result ZIP or folder; leave empty for a fresh run")
    base.RUN = base.RUN.replace(
        "_five_factor_ablations'", "_five_factor_ablations_recovery_v2'")
    base.RUN = base.RUN.replace(
        "print('Running six paired variants:', run_name, flush=True)",
        "if MIGRATION_SOURCE:\n"
        "    from ablation_runner import migrate_saved_results\n"
        "    imported = migrate_saved_results(\n"
        "        MIGRATION_SOURCE, output, dataset, configured,\n"
        "        PROJECT_ROOT / 'data/label_corrections.json', SEED, LIMIT)\n"
        "    print('Imported saved method results:', imported['imported_method_rows'])\n"
        "print('Running six paired variants:', run_name, flush=True)")
    base.INTRO += (
        '\n**Recovery revision 2:** If Ollama aborts a request with its token-repeat\n'
        'limit, this runner logs the failure and tries a different seed and shorter\n'
        'generation. It permits at most four attempts for a logical call across\n'
        'resumptions. Failed attempts have unknown token costs. The search method\n'
        'and five priority ablations are unchanged. This notebook uses a new output\n'
        'directory and does not silently mix with results from the previous runner.\n\n'
        'To reuse unaffected saved results, set `MIGRATION_SOURCE` to the old ZIP\n'
        'or results folder. The importer checks code, data, labels, model config,\n'
        'seed, and question IDs, and records the old runner hash. A clean final study\n'
        'should still be repeated entirely under one frozen version.\n\n'
        'For the saved 50-question development run, keep `SPLIT="dev"`, `LIMIT=50`,\n'
        '`SEED=42`, and `MODEL="qwen2.5:7b"`. Point `MIGRATION_SOURCE` to the old\n'
        'results folder if it remains in `/kaggle/working`, or upload the partial\n'
        'ZIP as a Kaggle input and use its ZIP path.\n'
    )
    return base.build()


if __name__ == '__main__':
    build()
