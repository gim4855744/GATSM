# Generalized Additive Time Series Model (GATSM)

[![AAAI](https://img.shields.io/badge/AAAI-2026-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/41084)
[![ArXiv](https://img.shields.io/badge/ArXiv-2410.10535-red)](https://arxiv.org/abs/2410.10535)

A PyTorch implementation for Generalized Additive Time Series Model (GATSM), proposed in the paper "*Transparent Networks for Multivariate Time Series*".

## Requirements
- torch == 2.0.1
- lightning == 2.2.4
- tsai == 0.3.9
- scikit-learn == 1.4.2
- PyYAML == 6.0.1

## Dataset directory

### For the PhysioNet datasets
Please refer the PhysioNet web site.
- Challenge 2012 (Mortality): https://physionet.org/content/challenge-2012/1.0.0/
- Challenge 2019 (Sepsis): https://physionet.org/content/challenge-2019/1.0.0/

### For the UCR and Monash datasets
The datasets will be automatically downloaded using the tsai library. If tsai does not download datasets, please manually download them with the following directory structure.
```
./data
L<dataset1>
 L<dataset files>
L<dataset2>
 L<dataset files>
...
```

## Hyper-parameter tuning
```bash
python hparams/tune_gatsm.py --dataset=<dataset name>
```

## Training
```bash
python run_experiments.py --mode=train --dataset=<dataset name> --seed=<random seed>
```

## Testing
```bash
python run_experiments.py --mode=test --dataset=<dataset name> --seed=<random seed>
```

## Visualization
```bash
python visualization.py --dataset=<dataset name> --seed=<random seed>
```

## Citation
```bibtex
@inproceedings{kim2026transparent,
    title={Transparent Networks for Multivariate Time Series},
    author={Kim, Minkyu and Lee, Suan and Kim, Jinho},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2026},
}
```
