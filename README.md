# Fake News Detection using LSTM

This project implements an LSTM-based deep learning model to classify news articles as "fake" or "real."

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)

---

## Description

The goal is to detect fake news using NLP and an LSTM neural network. The workflow includes:
- Text preprocessing (cleaning, tokenization, lemmatization, stopword removal).
- Sequence tokenization and padding.
- Training an LSTM model with embedding and dense layers.
- Evaluation via accuracy/loss plots and confusion matrix.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HemantAjmer/fake-news-detection.git
   cd fake-news-detection
2. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn nltk tensorflow scikit-learn
3. **Download NLTK resources (run in Python)**:
   ```python
   import nltk
   nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'])
   
## Dataset
  * Fake News: Fabricated articles (label = 0).
  * True News: Genuine articles (label = 1).
  *  Columns used: text (article content) and label.

## Model Architecture
    ``` Python
    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_1 (InputLayer)        [(None, 150)]             0         
                                                                     
     embedding (Embedding)       (None, 150, 128)          6400000   
                                                                     
     dropout (Dropout)           (None, 150, 128)          0         
                                                                     
     lstm (LSTM)                 (None, 150, 128)          131584    
                                                                     
     global_max_pooling1d (Glob  (None, 128)               0         
     alMaxPooling1D)                                                 
                                                                     
     dense (Dense)               (None, 64)                8256      
                                                                     
     dropout_1 (Dropout)         (None, 64)                0         
                                                                     
     dense_1 (Dense)             (None, 2)                 130       
                                                                     
    =================================================================
    Total params: 6,539,970
    Trainable params: 6,539,970
    Non-trainable params: 0
    _________________________________________________________________
  * Optimizer: Adam (learning_rate=0.0005).
  * Loss: Categorical cross-entropy.
## Results
  * Training Accuracy: ~99% after 10 epochs.
  * Validation Accuracy: ~98%.
