# CS4248 Team29: Labeled Unreliable News
## Abstract
Today, generative artificial intelligence (AI) is highly capable of automating content creation, thereby driving the widespread proliferation of indistinguishable fake news at a rate that overwhelms fact-checkers. This study developed a system to predict whether a news article or statement, specifically political ones, should be classified as trusted or misinformation. The system employs a logistic regression model to predict directly extractable features and derived features from a DistilBERT model. Detailed analysis was conducted to comprehend the impact each feature has on the model's prediction.

## Folder Structure
```
cs4248-team29-Labeled-Unreliable-News/
│
├── analysis_code/ # Directory for analyzing the impact of Valence-Arousal-Dominance (VAD)
│ ├── log_emo.py # Code for logistic regression on emotional analysis
│ ├── vad_plot.py # Code to generate 3D cluster plot using VAD values
│ ├── bert_features_correlation.ipynb # Code for finding BERT features correlation to manual engineered features
│ └── model_comparison.ipynb # Code for model performance comparison
│
├── data_code/ # Directory for data manipulation and preprocessing
│ ├── combine_data.py # Code to combine different datasets
│ ├── fix_data.py # Code to clean and fix issues in datasets
│ ├── permute_data.py # Code to permute data for model robustness
│ └── resize_data.py # Code to resize datasets accordingly
│
├── model_code/ # Directory related to the modeling aspect of the project
│ ├── hiddenState.py # Code to train and extract hidden states
│ ├── model1.py # Code for model1
│ └── model2.py # Code for model2
│
├── .gitignore # Specifies intentionally untracked files to ignore
├── README.md # README file with project information and instructions
├── baseline_log_reg_model.ipynb # Jupyter notebook for baseline logistic regression model
└── distilbert_classifier.ipynb # Jupyter notebook for DistilBERT classification model
```

## Dataset
The dataset used for training includes data from multiple sources that have been preprocessed to fit the needs of our analysis. These sources include:
- CompareNet
- ISOT
- LIAR

## Setup
This project uses Python 3.10.13.<br>
Run `pip install -r requirements.txt` to install the required dependencies before executing the Python code.

## Acknowledgments
- Mentor: Rishabh Anand
- Team Members: A0219673W, A0218514J, A0219814B, A0201348N, A0280003Y
