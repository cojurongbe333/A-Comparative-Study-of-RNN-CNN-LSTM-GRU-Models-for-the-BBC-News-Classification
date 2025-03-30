# BBC News Classification with Enhanced RNN & Explainability

This repository contains a full pipeline for classifying BBC news articles into predefined categories using an enhanced Bidirectional RNN model. The project applies advanced NLP preprocessing, feature engineering, explainability tools (LIME/SHAP), and performance visualization.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Explainability](#explainability)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Authors](#authors)

## Overview

The goal of this project is to develop a robust and interpretable deep learning model that accurately classifies news articles from the BBC dataset into five categories:
- **Business**
- **Entertainment**
- **Politics**
- **Sport**
- **Tech**

Using techniques like synonym replacement, paraphrasing, TF-IDF prioritization, and advanced RNN modeling, we achieved strong classification performance while making the model more explainable using LIME and t-SNE.

## Dataset

The dataset used is the **BBC Text Classification Dataset**, which contains short news articles labeled with one of five categories.

- Format: CSV
- Columns: `category`, `text`
- Source: [BBC News Dataset](http://mlg.ucd.ie/datasets/bbc.html)

## Preprocessing

The preprocessing pipeline includes:

- **Text Cleaning**: Lowercasing, removing numbers/special characters, and filtering out custom stopwords.
- **TF-IDF + Chi-Squared**: Extract top features important for classification.
- **Feature Prioritization**: Repetition boosting for top features.
- **Data Augmentation**:
  - **Synonym Replacement** (WordNet-based, only on high-impact words)
  - **Random Word Drop Paraphrasing**
- **LIME & SHAP**: Used to improve interpretability and guide data augmentation.
- **Tokenization & Padding**: Using Keras `Tokenizer` and `pad_sequences`.

## Model Architecture

A deep learning model using Keras Sequential API:
- **Embedding Layer**: 300-dimensional word embeddings
- **Bidirectional SimpleRNN**: Captures context in both directions
- **Global Max Pooling**: Reduces dimensionality
- **Layer Normalization**: For stable convergence
- **Dense Layers with Dropout and L2 Regularization**
- **Softmax Output**: Multi-class prediction

**Loss**: Sparse Categorical Crossentropy  
**Optimizer**: Adam (with learning rate scheduler)  
**Callbacks**: EarlyStopping and ReduceLROnPlateau  

## Performance

- **Accuracy**: 91%
- **Macro F1 Score**: 90%
- **Per-Class F1 Scores**:
  - Tech: 0.94
  - Business: 0.95
  - Sport: 0.97
  - Entertainment: 0.82
  - Politics: 0.84

Visualizations include:
- Accuracy/Loss curves  
- Confusion matrix  
- Precision-Recall curves  
- t-SNE plot for class separation

## Explainability

- **LIME**: Highlights influential words in classification decisions.
- **SHAP**: (Experimental) attempted to explain sparse vector importance.
- **t-SNE**: Visualized feature embeddings showing clear class separability.

## Installation

```bash
git clone https://github.com/yourusername/bbc-news-classifier.git
cd bbc-news-classifier
pip install -r requirements.txt
```

Requirements include:
- `tensorflow`
- `scikit-learn`
- `nltk`
- `shap`
- `lime`
- `matplotlib`
- `seaborn`
- `pandas`

## Usage

1. Place the `bbc_text.csv` file in the root directory.
2. Run the notebook:
   ```bash
   jupyter notebook DL_Group_Project_BBC.ipynb
   ```
3. The notebook will:
   - Preprocess the dataset
   - Train the RNN model
   - Visualize performance
   - Run LIME explanations

## Results

The enhanced preprocessing and augmentation strategies significantly improved model generalization, especially for underperforming categories like **business** and **entertainment**.

Final accuracy: **91%**  
Model shows strong precision and recall across all categories.

## Authors

- Chantal Ojurongbe
- Bess Shebeck
- Zachary Tisdale
- Zhongyi Sun

Contributions: Preprocessing, modeling, augmentation, evaluation, and explainability.

## License

MIT License - see the [LICENSE](LICENSE) file for details.
