"""Study constants from the original project (Nuri Korhan and coauthors).

Repository packaging: GPT-6 Astra / Codex, 2026-09-25. Paths are supplied by
the caller; no user-specific machine or manuscript-production paths are used.
"""
from pathlib import Path

RANDOM_SEED = 1
SIZE_OF_WALSH = 16
KERNEL_SIZE = 5
BATCH_SIZE = 16
DATASETS = {
    "D1": {"name": "BCI IV-2a", "moabb_class": "BNCI2014_001", "n_subjects": 9,
           "n_classes": 4, "n_channels": 22, "fs": 250, "pickle_prefix": "D1_BCIIV2a"},
    "D3": {"name": "BNCI2014-002", "moabb_class": "BNCI2014_002", "n_subjects": 14,
           "n_classes": 2, "n_channels": 15, "fs": 512, "pickle_prefix": "D3_BNCI2014_002"},
    "D4": {"name": "BNCI2014-004", "moabb_class": "BNCI2014_004", "n_subjects": 9,
           "n_classes": 2, "n_channels": 3, "fs": 250, "pickle_prefix": "D4_BCIIV2b"},
}


def data_path(work_dir, dataset, subject, *, raw=False):
    folder = "Raw_Datasets" if raw else "CSP_Datasets"
    suffix = ".pkl" if raw else ".pickle"
    return (Path(work_dir) / "Pickle_All_Datasets" / folder /
            f"{DATASETS[dataset]['pickle_prefix']}_S{subject:02d}{suffix}")


def selected_subjects(dataset, subjects=None):
    valid = range(1, DATASETS[dataset]["n_subjects"] + 1)
    if subjects is None:
        return list(valid)
    invalid = sorted(set(subjects) - set(valid))
    if invalid:
        raise ValueError(f"Invalid subjects for {dataset}: {invalid}")
    return [s for s in valid if s in subjects]
