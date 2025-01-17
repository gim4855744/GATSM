# Generalized Additive Time Series Model (GATSM)

A PyTorch implementation for Generalized Additive Time Series Model (GATSM), proposed in the paper "<em>Transparent Networks for Multivariate Time Series<em>".

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
@misc{kim2024transparent,
      title={Transparent Networks for Multivariate Time Series}, 
      author={Minkyu Kim and Suan Lee and Jinho Kim},
      year={2024},
      eprint={2410.10535},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.10535}, 
}
```