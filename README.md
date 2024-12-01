# LLM Spam Classifier

Welcome to the **LLM Spam Classifier** repository! This project demonstrates how to classify SMS messages as spam or not spam using a pre-trained GPT-2-based language model. 
The dataset is sourced from the UCI SMS Spam Collection dataset. 

---

## Features
- Data preprocessing: Balances the dataset for spam and ham messages.
- Fine-tunes a GPT-based model for classification tasks.
- Supports evaluation of the model on training, validation, and test datasets.
- Provides utilities for inference and accuracy evaluation.

---

## Requirements
1. Python 3.8+
2. Git
3. Pip for package management
4. pip install -r requirements.txt
chainlit==1.3.2
tiktoken==0.7.0
tokenizers==0.19.1

---

## Setup Instructions

### Clone the Repository
```bash
git clone https://github.com/silsgah/spamClassifier.git
cd spamClassifier
## Usage
Step 1: Data Preparation
The dataset is automatically downloaded, extracted, and preprocessed.
Run the following command:
bash
Copy code
This script:
1. Downloads the SMS Spam Collection dataset.
2. Balances the dataset (equal spam and ham messages).
3. Splits the dataset into training, validation, and test sets.
4. Tokenizes the text for model consumption.
###Step 2: Training the Classifier
The training script fine-tunes a GPT-based model to classify messages as spam or ham.
To train the model:  main.py
###Step 4: Make Predictions
from classifier import classify_review

text_1 = "You are a winner! You have been selected to receive $1000 cash."
prediction = classify_review(text_1, model, tokenizer, device)
print(f"Prediction: {prediction}")
### Folder Structure
spamClassifier/
├── data/                 # Dataset folder
├── main.py               # Main script for data preparation
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── predict.py            # Prediction script
├── classifier/           # Core functionality (model, dataset, utility scripts)
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
### Sample results
Starting training...
Ep 1 (Step 000000): Train loss 2.153, Val loss 2.392
Ep 1 (Step 000050): Train loss 0.617, Val loss 0.637
Ep 1 (Step 000100): Train loss 0.523, Val loss 0.557
Training accuracy: 70.00% | Validation accuracy: 72.50%
Ep 2 (Step 000150): Train loss 0.561, Val loss 0.489
Ep 2 (Step 000200): Train loss 0.419, Val loss 0.397
Ep 2 (Step 000250): Train loss 0.409, Val loss 0.353
Training accuracy: 82.50% | Validation accuracy: 85.00%
Ep 3 (Step 000300): Train loss 0.333, Val loss 0.320
Ep 3 (Step 000350): Train loss 0.340, Val loss 0.306
Training accuracy: 90.00% | Validation accuracy: 90.00%
Ep 4 (Step 000400): Train loss 0.136, Val loss 0.200
Ep 4 (Step 000450): Train loss 0.153, Val loss 0.132
Ep 4 (Step 000500): Train loss 0.222, Val loss 0.137
Training accuracy: 100.00% | Validation accuracy: 97.50%
Ep 5 (Step 000550): Train loss 0.207, Val loss 0.143
Ep 5 (Step 000600): Train loss 0.083, Val loss 0.074
Training accuracy: 100.00% | Validation accuracy: 97.50%
Training completed in 13.93 minutes.
Final Training Accuracy: 100.00%
Final Validation Accuracy: 97.50%
Final Training Loss: 0.083
Final Validation Loss: 0.074
Training accuracy: 97.21%
Validation accuracy: 97.32%
Test accuracy: 95.67%
Text 1 Classification: spam
Text 2 Classification: not spam
###License
This project is licensed under the MIT License. See the LICENSE file for details.



