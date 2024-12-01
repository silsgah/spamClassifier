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
###License
This project is licensed under the MIT License. See the LICENSE file for details.



