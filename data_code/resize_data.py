"""
Author: Daren Tan (Team 29)
Date: Mar 27, 2024
Description: Python script to split train set output from combine_data.py into train-val-test set

1) This code is a modification to fix_data.py where sentences are split as well
2) This code should be executed after combine_data.py, since it's meant to fix dataset generated from that
"""

import os
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split

# Download nltk data
nltk.download('punkt')

def get_original_df():
    """
    Load data that was originally designated as training set in combine_data.py

    :return: DataFrame of the training set
    """
    file_path = os.path.join(os.getcwd(), "dataset", "combine_data", "train.csv")
    df = pd.read_csv(file_path)
    
    return df


def sent_split_df(df):
    """
    For each document in the given DataFrame, split them into sentences corresponding to an entry in the new DataFrame

    :param df: DataFrame containing documents with multiple sentences
    :return New DataFrame where "Document" column contains one sentence each
    """
    # Define the function to split each "Document" row of the original DataFrame
    def split_sent_keep_label(text, label, dataset):
        sentences = sent_tokenize(text)
        return [{"Document": sentence, "Label": label, "Dataset": dataset} for sentence in sentences]
    
    # Create a new DataFrame from the split rows
    new_rows = []
    for index, row in df.iterrows():
        sentences_and_labels = split_sent_keep_label(row["Document"], row["Label"], row["Dataset"])
        new_rows.extend(sentences_and_labels)
    df = pd.DataFrame(new_rows)
    
    # Filter out sentences with:
    # (1) Less than 5 words (Not much semantic information can be learnt from short sentences)
    # (2) or more than 512 words (BERT has a maximum token limit of 512 tokens per sequence)
    df = df[df["Document"].apply(lambda x: len(x.split()) >= 5 and len(x.split()) <= 512)]

    # Remove duplicates from dataset (if any)
    df = df.drop_duplicates(subset=["Document"], keep='first')

    return df


def dataset_split_df(df):
    """
    Split the dataset into training, validation, and testing sets

    :param df: DataFrame where each "Document" holds one sentence
    """
    # Perform train-test split to generate train, validation, and test sets
    train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df[["Label", "Dataset"]], random_state=4248)
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df[["Label", "Dataset"]], random_state=4248)

    # Define and create output directory if does not exist
    out_directory = os.path.join(os.getcwd(), "dataset", "fixed_data_2")
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    # Save the various DataFrames sets into CSV files
    train_df.to_csv(os.path.join(out_directory, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out_directory, "test.csv"), index=False)
    val_df.to_csv(os.path.join(out_directory, "val.csv"), index=False)
    train_val_df.to_csv(os.path.join(out_directory, "train_val.csv"), index=False)


if __name__ == '__main__':
    original_df = get_original_df()
    processed_df = sent_split_df(original_df)
    dataset_split_df(processed_df)
