"""
Author: Daren Tan (Team 29)
Date: Apr 09, 2024
Description: Python script to (1) Masked NEs in current dataset; (2) Re-split dataset using their source

1) This code should be executed after resize_data.py, since it uses the data generated from that script
"""

import os
import spacy
import pandas as pd

def read_data():
    """
    Read the original train and test data

    :return: DataFrame of the original train data
             DataFrame of the original test data
    """
    data_directory = os.path.join(os.getcwd(), "dataset/fixed_data_2")

    train_df = pd.read_csv(os.path.join(data_directory, "train_val.csv"))
    test_df = pd.read_csv(os.path.join(data_directory, "test.csv"))

    return train_df, test_df


def create_output_directory():
    """
    Create the directory to save DataFrames

    :return: Path to output directory
    """
    directory = os.path.join(os.getcwd(), "permutated_datasets")
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def generate_masked_data(train_df, test_df, out_dir):
    """
    Masked the Named Entities in the datasets
    The goal is to see how much does NEs affect the model's prediction

    :param train_df: DataFrame of the original train data
    :param test_df:  DataFrame of the original test data
    :param out_dir:  Output directory path
    """
    nlp = spacy.load("en_core_web_sm")

    def replace_named_entities(text):
        doc = nlp(text)
        for ent in doc.ents:
            text = text.replace(ent.text, '<NE>')
        return text

    train_copy_df = train_df.copy()
    test_copy_df = test_df.copy()

    train_copy_df['Document'] = train_copy_df['Document'].apply(replace_named_entities)
    test_copy_df['Document'] = test_copy_df['Document'].apply(replace_named_entities)

    train_copy_df.to_csv(os.path.join(out_dir, "train1.csv"), index=False)
    test_copy_df.to_csv(os.path.join(out_dir, "test1.csv"), index=False)


def separate_data_by_source(train_df, test_df, out_dir):
    """
    Reorganize the train and test sets by dataset they originated from:
    - train: 1  test: 2 and 3   (260344 246201)
    - train: 2  test: 1 and 3   (237397 269148)
    - train: 3  test: 1 and 2   (8804 497741)
    The goal is to see how well the model generalizes to other datasets

    :param train_df: DataFrame of the original train data
    :param test_df:  DataFrame of the original test data
    :param out_dir:  Output directory path
    """
    
    df = pd.concat([train_df, test_df], ignore_index=True)

    df_datasets = dict()
    for k, v in df.groupby("Dataset"):
        df_datasets[k] = v

    partitions = [[1, (2, 3)], [2, (1, 3)], [3, (1, 2)]]

    for i, partition in enumerate(partitions):
        new_train_df = df_datasets[partition[0]]
        new_test_df = pd.concat([df_datasets[partition[1][0]], df_datasets[partition[1][1]]], ignore_index=True)

        new_train_df.to_csv(os.path.join(out_dir, f"train{i + 2}.csv"), index=False)
        new_test_df.to_csv(os.path.join(out_dir, f"test{i + 2}.csv"), index=False)

        print(new_train_df.shape[0], new_test_df.shape[0])


if __name__ == '__main__':
    train_df, test_df = read_data()
    out_dir = create_output_directory()
    generate_masked_data(train_df, test_df, out_dir)
    separate_data_by_source(train_df, test_df, out_dir)
