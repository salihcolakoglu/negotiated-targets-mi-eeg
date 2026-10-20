# EEG dataset attribution and access

The original experiments reanalyse three public motor imagery EEG datasets; the separate extensions add Lee/OpenBMI and SHU. The [BNCI Horizon 2020 catalogue](https://bnci-horizon-2020.eu/database/data-sets) provides the recordings and their terms. The preparation scripts download the recordings through MOABB.

| BNCI identifier | Study label | Dataset description | Related publication linked by BNCI |
|---|---|---|---|
| BNCI2014-001 | D1 | [BCI Competition IV dataset 2a](https://www.bbci.de/competition/iv/desc_2a.pdf), Brunner et al. (2008) | Tangermann et al. (2012), [Review of the BCI Competition IV](https://doi.org/10.3389/fnins.2012.00055) |
| BNCI2014-002 | D3 | [Two-class motor imagery dataset](https://bnci-horizon-2020.eu/database/data-sets) | Steyrl et al. (2016), [Random forests in non-invasive sensorimotor rhythm brain–computer interfaces: A practical and convenient non-linear classifier](https://doi.org/10.1515/bmt-2014-0117) |
| BNCI2014-004 | D4 | [BCI Competition IV dataset 2b](https://www.bbci.de/competition/iv/desc_2b.pdf), Leeb et al. (2008) | Leeb et al. (2007), [Brain-computer communication: motivation, aim, and impact of exploring a virtual apartment](https://doi.org/10.1109/TNSRE.2007.906956) |

The catalogue names the **Institute for Knowledge Discovery, Graz University of Technology** as licensor and **[CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/)** as the license for all three datasets. Retain this credit, the license information and the relevant source publications when sharing source dataset content. The dataset descriptions and related publications are separate sources.

## Shared results and source labels

The original archive includes preparation, training and analysis software, numerical summaries, and 800 EEGNet prediction files. Additional archives have their own inventories and dataset roles, described in `extensions/README.md`. It does not include raw EEG or filtered and epoched signal caches. Preparation writes downloaded recordings to `data/` and generated signal inputs to `work/`; neither directory is part of the release.

Within `reference/stored_results.zip`, prediction files contain model outputs (`P_train`, `P_test`), and Walsh conditions also contain the last targets used for training (`Y_used_last`). They additionally contain the source datasets' class labels (`y_train`, `y_test`), encoded as integer class indices. The labels remain source dataset content under the upstream terms; the authors do not relicense them under MIT or CC BY 4.0.

The authors' software uses MIT. Their original documentation and generated predictions, targets and numerical summaries use CC BY 4.0, to the extent copyright or similar rights apply. See [Licensing](LICENSING.md) for the scope of each license. This distinction does not change the source datasets' licenses or describe every computed result as an adaptation of a copyrighted recording.

Please cite the EEG study as well as the original data sources when using the methods or results, and explain changes that affect reproduction. Academic citation requests do not add conditions to the MIT software license.

## Additional data sources

Lee/OpenBMI source data are available at [10.5524/100542](https://doi.org/10.5524/100542), under CC0-1.0. The source paper is Lee et al. (2019), [EEG dataset and OpenBMI toolbox for three BCI paradigms](https://doi.org/10.1093/gigascience/giz002). The study uses the offline calibration block from the first and second recording days; the source variable called `EEG_MI_test` is an online block and is not used.

SHU source data are the [Figshare version 1 MAT release](https://doi.org/10.6084/m9.figshare.19228725.v1), under CC BY 4.0. The associated paper is Ma et al. (2022), [A large EEG dataset for studying cross-session variability in motor imagery brain-computer interface](https://doi.org/10.1038/s41597-022-01647-1). This specific version is distinct from later EDF/NEMAR releases. Source event order is preserved, including the 99-trial subject001/day4 recording.

Neither additional raw source nor prepared EEG arrays are redistributed. Their class labels and attributed event metadata retain the provider terms above. Prepared-input training contracts identify the fitting, reserved first-day confidence and later-day test roles explicitly. The reference predictions cannot be treated as new raw EEG.
