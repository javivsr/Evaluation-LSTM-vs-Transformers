# Extrapolating Neural Calculator: Evaluating LSTM vs. Transformer Architectures

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📄 Overview
This repository contains the full implementation for the **Machine Learning Final Project (University of Bologna - UniBo)** under the **Full Practical Track**. 

The goal of this project is to build and evaluate an **extrapolating neural calculator** capable of receiving mathematical expressions as raw strings (e.g., `9-(4-(1+2))`) and predicting their integer output values[cite: 3]. The primary challenge is evaluating generalization beyond the training distribution—specifically testing robustness against unusual formula shapes, deeper nesting, different subtraction patterns, and **longer sequence lengths** than those seen during training.

## ⚙️ Key Constraints & Rules
- **Strict Data Augmentation Rule:** No example used for training or validation exceeded 21 characters in length.
- **No Direct Symbolic Evaluation:** No use of Python's `eval`, custom parsers, or hardcoded reduction algorithms. All predictions are generated solely by learned neural network architectures.

## 🧪 Models Implemented
We implemented and compared two substantially different deep learning architectures:
1. **Bidirectional LSTM with Attention Pooling / Sequence-to-Sequence LSTM**: Captures sequential dependencies and hierarchical structure in arithmetic expressions.
2. **Transformer Encoder / Sequence-to-Sequence Transformer**: Utilizes multi-head self-attention mechanisms and positional encodings to model long-range dependencies and complex operator syntax.

## 📊 Performance & Results
Models were evaluated across four distinct splits using **Exact-Match Accuracy (EM)**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**:

| Model Architecture | In-Distribution (ID) EM | Out-of-Distribution (OOD) EM | Long-Sequence Test EM | Target Threshold Status |
| :--- | :---: | :---: | :---: | :---: |
| **Minimum Required Threshold** | **≥ 50.0%** | **≥ 40.0%** | **≥ 25.0%** | **Passed** |
| **Bidirectional LSTM** | *99.89%* | *99.25%* | *89.67%* | ✅ Passed |
| **Transformer Encoder** | *87.76%* | *87.3%* | *51.12%* | ✅ Passed |


## 🚀 How to Run the Code

### Google Colab (Recommended)
1. Open the file `ML_Neural_Calculator_Project.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Select a **T4 GPU** runtime instance (`Runtime` > `Change runtime type` > `T4 GPU`).
3. Execute all cells sequentially. Pre-trained weights are automatically loaded via `gdown` to guarantee execution within 10 minutes.

### Local Setup
```bash
# 1. Clone the repository
git clone [https://github.com/javivsr/evaluacion-lstm-vs-transformers.git](https://github.com/javivsr/evaluacion-lstm-vs-transformers.git)
cd evaluacion-lstm-vs-transformers

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter Notebook
jupyter notebook ML_Neural_Calculator_Project.ipynb
```

## 👥 Authors & Contributions
- **Javier Vaca Sánchez** - Model architecture design, hyperparameter tuning, evaluation scripts, and LaTeX report preparation.
