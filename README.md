# Extrapolating Neural Calculator: Evaluating LSTM vs. Transformer Architectures

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

We learn a neural calculator that maps an arithmetic expression — single digits, the
operators `+` and `-`, and parentheses — to its integer value, **without any symbolic
evaluation**. The task is framed as scalar regression on a character sequence, and the
central challenge is **extrapolation**: performing correctly on longer and more deeply
nested expressions than those seen during training.

Two architecturally distinct models are compared: a unidirectional **LSTM** with
last-token read-out, and a **Transformer** encoder with mean pooling. Both clear all
required thresholds, but they fail for different reasons.

## The problem

A string over the alphabet `[0-9 + - ( )]` maps to a signed integer in the range
`[-99, 99]`. Every operand is a single digit. The model may not evaluate anything
symbolically or use hand-crafted arithmetic rules — it must learn the computation
exclusively from examples.

Training and validation expressions are short and structurally limited. Evaluation adds
two harder splits: an **out-of-distribution (OOD)** set with different structural
properties, and a **long-sequence** set with substantially longer formulas. Exact-match
accuracy rounds the prediction to the nearest integer.

## Models

| | LSTM | Transformer |
| --- | --- | --- |
| Configuration | 1 layer, unidirectional, hidden 256, embedding 16, dropout 0.1 | 4 encoder layers, 4 heads, model dim. 128, FFN 384, dropout 0.1 |
| Read-out | Last hidden state (pre-padding so the last position is the last real character) | Mean pooling over non-padded token representations |
| Motivation | Arithmetic accumulates left to right, so the final hidden state already encodes the running result | Self-attention links non-local structure, which nested parentheses require |

Design decisions were reached after discarding alternatives: a bidirectional LSTM with
attention pooling and a GRU both extrapolated worse; for the Transformer, a special
end-of-sequence token, last-token read-out and max pooling were all tried before mean
pooling.

## Training

- **Data:** 12,000 training examples, 2,000 validation.
- **Augmentation:** 6,000 synthetic expressions generated from the project grammar — deep
  chains to increase nesting depth, and broad expressions for structural variety. Applied
  to the Transformer, which needed it; the LSTM already generalised and trained on the
  original data.
- **Optimisation:** AdamW, learning rate 5e-4, weight decay 1e-2, Smooth L1 loss, batch
  size 32, gradient clipping at norm 1.0, 50 epochs. Cosine annealing for the LSTM,
  optional OneCycleLR warm-up for the Transformer.
- Both implemented in **PyTorch** with masked computation for padding.

## Results

Each architecture was trained over **multiple independent runs** with different random
seeds. No run was discarded. Because single-run figures can mislead, both the best run
and the across-run mean (with a 95% confidence interval) are reported.

### Mean exact-match accuracy over multiple runs

| Split | LSTM | Transformer |
| --- | --- | --- |
| Train | 99.86% | 88.82% |
| Validation | 99.12% | 90.39% |
| Test In-distribution | 99.10% | 90.16% |
| Test OOD | **95.50%** | 80.91% |
| Test Long sequences | **83.77%** | **53.80%** |

### Best run — LSTM

| Split | EM (%) | MAE | RMSE |
| --- | --- | --- | --- |
| Test In | 99.85 | 0.0388 | 0.0627 |
| Test Out | 99.20 | 0.0609 | 0.1367 |
| Test Long | 98.07 | 0.1078 | 0.3857 |

### Best run — Transformer

| Split | EM (%) | MAE | RMSE |
| --- | --- | --- | --- |
| Test In | 95.25 | 0.1773 | 0.3880 |
| Test Out | 88.10 | 0.3437 | 1.1057 |
| Test Long | 61.80 | 1.3429 | 3.3385 |

## Key finding: the two models fail for different reasons

This is the most interesting result of the project. Grouping per-example absolute error on
the long-sequence split along four axes (sequence length, operator count, parenthesis
depth, and result magnitude) shows two **opposite** failure profiles:

- **The LSTM is limited numerically.** Its error is essentially flat across sequence
  length, operator count and nesting depth, but jumps as the magnitude of the result
  grows — from roughly 0.4–0.6 MAE for results up to 49, to about 2.7 for results in
  `[50, 99]`. It learned to combine a bounded number of single-digit terms regardless of
  expression shape, and only collapses outside the densely sampled value range.
- **The Transformer is limited structurally.** Its error grows monotonically with
  sequence length (from ≈0.4 at length 15 to ≈9 at length 27), with operator count and
  with parenthesis depth, while result magnitude barely affects it. It is the classic
  difficulty of extrapolating positional structure beyond the training range.

Since the long-sequence split introduces structural rather than numeric complexity, the
LSTM's inductive bias wins in exactly the regime that matters — which is why the gap
widens to 83.77% against 53.80% there.

Variability also concentrates under extrapolation: the LSTM's 95% CI is within about ±0.5
points in-distribution but spans roughly ±6.4 points on the long split. The realistic
figure to trust is the mean, not the peak.

## What I would do differently

- The LSTM's numeric limit would need targeted augmentation towards large-magnitude
  targets, which the 21-character budget makes hard to manufacture.
- The Transformer's positional encoding would benefit from progressive training, growing
  length and complexity gradually.
- Hyperparameters were selected by manual tuning rather than a systematic search.

## How to run

```shell
git clone [github.com](https://github.com/javivsr/Evaluation-LSTM-vs-Transformers.git)
cd Evaluation-LSTM-vs-Transformers
pip install -r requirements.txt
jupyter notebook ML_Neural_Calculator_Project.ipynb
```
## Repository structure

```
ML_Neural_Calculator_Project.ipynb   # data generation, training and evaluation pipeline
requirements.txt                     # dependencies
```



## Context

Final project for the Machine Learning 2025/26 course, Full Practical Track, during the Erasmus+ exchange year at Alma Mater Studiorum – Università di Bologna.

Authors: Javier Vaca, Gabriele Toro, Marcos García.

My contribution: design of both model architectures, the overall training procedure, hyperparameter tuning, and the LaTeX report — including the extrapolation analysis and experimental comparison as a team effort.

## References
- Hochreiter & Schmidhuber, Long Short-Term Memory, Neural Computation, 1997
- Vaswani et al., Attention Is All You Need, NeurIPS, 2017
- Trask et al., Neural Arithmetic Logic Units, NeurIPS, 2018
