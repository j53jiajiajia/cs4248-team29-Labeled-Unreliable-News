"""
Author: Daren Tan (Team 29)
Date: Apr 05, 2024
Description: Python script to analyse impact of Emotions on the Classification using LogReg

References:
- VAD Function: https://stackoverflow.com/questions/63831591/how-to-find-valence-arousal-dominance-of-a-text-tweet-using-any-python-sent
- VAD Scores: https://github.com/bagustris/text-vad/blob/master/VADanalysis/lib/vad-nrc.csv
"""

import os
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression

# Download NLTK data
nltk.download('punkt')

DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), "data_code/dataset")

def read_files():
    """
    Read the data files and the file containing annotated VAD values for text

    :return: DataFrame of the training + validation data
             VAD scores containing a word and its corresponding VAD values
    """
    train_val_df = pd.read_csv(os.path.join(DATA_DIR, "fixed_data_2/train_val.csv"))
    vad_scores = pd.read_csv(os.path.join(DATA_DIR, "emotion_data/vad-nrc.csv"), index_col="Word")

    return train_val_df, vad_scores


def VAD(text, vad_scores):
    """
    Compute VAD score of a text using words annotated with VAD values

    :param text:       String of a sentence
    :param vad_scores: VAD scores containing words and its corresponding VAD values
    :return:           1-D array-like object containing VAD value for the input text
    """
    i, j = 0, 0
    text_vad = np.zeros([3,])
    tokens = word_tokenize(text.lower())
    
    # Iterate through each token to sum up the VAD value
    for index, word in enumerate(tokens):
        neg = 1
        isNeg = False
        
        if word in vad_scores.index:
            # Invert word polarity if a negating word and/or contraction
            if index != 0:
                if tokens[index - 1] in ["no", "not"]:
                    isNeg = True
            if index != len(tokens) - 1:
                if tokens[index + 1] == "n\'t":
                    isNeg = True
            if isNeg:
                neg = -1

            text_vad = vad_scores.loc[word] * neg + text_vad
            i += 1
        j += 1

    # Handle case where no valid words are found
    if i == 0:
        return pd.Series([0, 0, 0])
    
    # Normalize the VAD values based on number of words found
    vad_values = [text_vad.valence/i, text_vad.arousal/i, text_vad.dominance/i]
    return pd.Series(vad_values)


def investigate_relationship(df, vad_scores):
    """
    Use Logistic Regression to analyse how VAD affects class predictions
    Recall that Label 0 and 1 refers to fake and real news respectively

    :param df:         DataFrame of the training + validation data
    :param vad_scores: VAD scores containing a word and its corresponding VAD values
    """
    
    # Extract VAD values from each document
    df[["Valence", "Arousal", "Dominance"]] = df["Document"].apply(lambda x: VAD(x, vad_scores))
    
    # Save then read file (since it takes a while to process)
    save_filepath = os.path.join(DATA_DIR, "train_and_vad")
    df.to_csv(save_filepath, index=False)
    df = pd.read_csv(save_filepath)

    # Extract X (features) and y (labels)
    X = df[["Valence", "Arousal", "Dominance"]]
    y = df["Label"]
    
    coefficients_sum = np.zeros((num_epochs, len(X.columns)))
    bias_sum = 0
    
    # Run multiple epochs of LogReg to get average weights and bias
    num_epochs = 10
    for epoch in range(num_epochs):
        model = LogisticRegression()
        model.fit(X, y)

        coefficients_sum[epoch, :] = model.coef_
        bias_sum += model.intercept_[0]
    
    # Compute average weights and bias
    average_coefficients = np.mean(coefficients_sum, axis=0)
    average_bias = bias_sum / num_epochs
    average_diff = np.abs(average_coefficients - average_bias)
    
    print("Average Coefficients (Weights):", average_coefficients)
    print("Average Baseline (Bias):", average_bias)
    print("Average Absolute Difference (Impact):", average_diff)

    # Code Output:
    # Average Coefficients (Weights): [-0.30210078 -2.51324179  2.41848702]
    # Average Baseline (Bias): 0.8792337726627919
    # Average Absolute Difference (Impact): [1.18133455 3.39247556 1.53925325]


if __name__ == '__main__':
    train_df, vad_scores = read_files()
    investigate_relationship(train_df, vad_scores)


"""
Explanation of Results

In terms of impact: Arousal > Dominance > Valence

Valence
- Negative weight: Increase in valence results in decrease in probability of real news
- More positive emotions -> Higher chance of fake news
- Possible explanation: Real news tend to be bad news?
- Lowest impact, implying that positive/negative emotions may not the most indicative of news type

Arousal
- Negative weight: Increase in arousal results in decrease in probability of real news
- More intense emotions -> Higher chance of fake news
- Possible explanation: Fake news tends to be more sensationalised (invoke extreme emotions)

Dominance
- Positive weight: Increase in dominance results in increase in probability of real news
- Increase in sense of control over emotion -> Higher chance of real news
- Possible explanation: Using words with power/confidence/persuasiveness would imply that you strongly believe in the news that you are reporting

Disclaimer
- VAD annotated to the words are usually human annotated
- For different datasets, the scales may differ, so annotation behaviour may differ as well
"""