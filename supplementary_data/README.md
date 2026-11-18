# Supplementary data files (manuscript version 1.13)

These three archives are the supplementary data files of the manuscript *Walsh-coded and negotiated targets for motor imagery EEG decoding: a controlled evaluation of accuracy and confidence* (S. Çolakoğlu, M. Onat, N. Korhan), version 1.13 of 18 November 2026. The manuscript's Data availability statement and Supplementary Information name them by these file names. They are published here because the journal's submission system accepts no data archives as supplementary material and asks for a repository link instead.

| File | Bytes | SHA-256 | Contents |
|---|---:|---|---|
| `Supplementary-Data-1-v1.13.zip` | 35,882,232 | `bb1b0d75b74c013a39616cf3e11bd76809917a053e8cfeac7da55895de45e3c5` | Participant-level records and stored predictions of Experiments 1–3 |
| `Supplementary-Data-2-v1.13.zip` | 33,236,052 | `30b325d692acf7f78a9c0edc69d5fbc7c6a536fcdc96116552bd73a2926689ac` | S1/W5 control, record-based summaries, classical-pipeline benchmark, W0/W1 resource summary, equivalence and sensitivity analysis |
| `Supplementary-Data-3-v1.13.zip` | 38,294,518 | `6ab22d72b68d81c44e897cca051ce35c645f05797f7e8cf3c7c3edd778e09c99` | Protocols, programs, records and stored test outputs of the three additional sensitivity analyses (Supplementary Section S19) |

Each archive starts with a `README.txt` that describes its folders and holds its own checksum manifest. The first archive also contains `verify_package.py`, which checks every file of that archive against its manifest; the second contains scripts that recompute the S1/W5 results, the classical-pipeline and resource summaries and the equivalence analysis from the stored records, without raw EEG, model weights, fitting or inference.

The archives contain no raw EEG, no prepared signal arrays and no model or optimizer checkpoints. Raw recordings remain available from their providers (see [`../DATA_ATTRIBUTION.md`](../DATA_ATTRIBUTION.md)).

Licensing follows the repository scope in [`../LICENSING.md`](../LICENSING.md): programs are MIT; the authors' documentation and generated numerical results are CC BY 4.0; stored source labels keep their upstream terms (BNCI labels CC BY-ND 4.0, Lee/OpenBMI labels CC0-1.0).

Check the archives after download, for example:

```bash
sha256sum -c supplementary_data/SHA256SUMS.txt
```

The archives are byte-identical to the files prepared for the journal submission; they were added in release v2.3.0 and are not modified by later commits.
