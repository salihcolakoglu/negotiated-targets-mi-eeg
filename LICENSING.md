# Licensing of the EEG code and results

This release contains software, the authors' documentation and generated results, and class labels from third-party datasets. The following licenses apply to those materials separately.

| Material | License and scope |
|---|---|
| Software in `src/`, `reproduce.py`, and accompanying software configuration and dependency specifications | MIT, as provided in [`LICENSE`](LICENSE). |
| The authors' original documentation in this release, and their generated predictions, training targets and numerical summaries in `reference/` | [CC BY 4.0](LICENSES/CC-BY-4.0.txt), to the extent copyright or similar rights apply. |
| BNCI2014-001, BNCI2014-002 and BNCI2014-004 recordings and source dataset content, including the `y_train` and `y_test` class-label arrays within the prediction files | The providers' [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/) terms remain unchanged. See [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md). Raw recordings and prepared signal caches are not included. |
| External software dependencies, linked papers and externally hosted materials | Their respective licenses and terms. A citation or link does not relicense them. |

The rights holders for the authors' original materials are **Salih Çolakoğlu, Mustafa Onat and Nuri Korhan**, copyright 2026. Identify these authors, link to the release and its license, and indicate modifications when sharing material under CC BY 4.0. Do not imply their endorsement of a modified work.

The MIT software license requires preservation of its copyright and permission notice. The request to cite the accompanying study and describe software changes is an academic request, not an additional condition of MIT. This release does not withdraw permissions already granted for material distributed under an earlier MIT release.

`reference/stored_results.zip` is a mixed collection: the authors' generated model outputs, targets and summaries use CC BY 4.0, while the included source-label arrays retain their upstream terms. The archive as a whole must not be represented as solely MIT or solely CC BY 4.0. The license summary in software or archive metadata does not override this file-level scope.

The full CC BY 4.0 legal text is included in [`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt), obtained from [Creative Commons](https://creativecommons.org/licenses/by/4.0/legalcode.txt). The upstream data license is available from [Creative Commons](https://creativecommons.org/licenses/by-nd/4.0/legalcode.en).
