# Generalized Additive Time Series Model (GATSM)

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