# Implementation of the paper: [Bayesian Generative Adversarial Nets with Dropout Inference](https://dl.acm.org/doi/10.1145/3430984.3431016)

Run file with the following configurations for each of the three synthetic data sets: 

python bdgan.py --dataset 2dgrid --train_iter 100000 --out_dir results

python bdgan.py --dataset 2dring --train_iter 100000 --out_dir results

python bdgan.py --dataset synth --train_iter 10000 --out_dir results
