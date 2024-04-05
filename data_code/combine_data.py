"""
Author: Daren Tan (Team 29)
Date: Mar 19, 2024
Description: Python script to combine various political fake news datasets into a CSV file for training

Dataset Acknowledgements
1) CompareNet - https://github.com/BUPT-GAMMA/CompareNet_FakeNewsDetection/releases/download/dataset/raw_data.zip
2) ISOT - https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/download?datasetVersionNumber=1
3) Liar - https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
"""

import os
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt")

def combine():
    data_dir = os.path.join(os.getcwd(), "dataset", "raw_data")
    
    # Get DataFrame from CSV/TSV files
    data1_df = get_data1_df(data_dir)
    data2_df = get_data2_df(data_dir)
    data3_df = get_data3_df(data_dir)
    test_df = get_data1_df(data_dir)

    # Concatenate DataFrames to get one DataFrame
    combined_df = pd.concat([data1_df, data2_df, data3_df], axis=0).reset_index(drop=True)
    
    # Get statistics from each DataFrame
    statistics_str = ("These are some statistics for the following datasets:\n"
                      "(1) CompareNet Training dataset (original)\n"
                      "(2) ISOT Fake News dataset \n"
                      "(3) LIAR dataset\n"
                      "(4) Combination of 1 + 2 + 3 Training dataset\n"
                      "(5) CompareNet Testing dataset (original)\n\n")
   
    df_list = [data1_df, data2_df, data3_df, combined_df, test_df]
    for i, df in enumerate(df_list):
        print(f"Extracting statistics for dataset {i + 1}")
        
        statistics_str += f"Statistics for dataset {i + 1}\n"
        statistics_str += get_statistics(df)

    return combined_df, test_df, statistics_str


def get_data1_df(dir):
    """
    Read in CompareNet (original) dataset as a DataFrame
    - Download URL: https://github.com/BUPT-GAMMA/CompareNet_FakeNewsDetection/releases/download/dataset/raw_data.zip
    - Only data from fulltrain.csv is used
    - From the data, only 2-"Hoax" and 4-"Reliable News" were considered

    :param dir: Path to directory for raw data
    :return:    DataFrame of original dataset
    """

    # Read in csv as Pandas DataFrame
    filepath = os.path.join(dir, "data1_fulltrain.csv")
    df = pd.read_csv(filepath, header=None, names=["Label", "Document"])
    
    # Filter out 1-Satire and 3-Propaganda, leaving behind 2-Hoax and 4-ReliableNews
    df = df[df["Label"].isin([2, 4])].reset_index(drop=True)
    
    # Relabel 2-Hoax as 0-FakeNews, and 4-ReliableNews as 1-RealNews
    df["Label"] = df["Label"].replace({2: 0, 4: 1})

    # Mark the data as belonging to dataset 1
    df["Dataset"] = 1

    return df


def get_data2_df(dir):
    """
    Read in ISOT Fake News dataset as a DataFrame
    - Download URL: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/download?datasetVersionNumber=1
    - From the Real-News, only data with subject "politicsNews" was extracted
    - From the Fake-News, only data with subject "politics" was extracted

    :param dir: Path to directory for raw data
    :return:    DataFrame of ISOT Fake News dataset
    """

    # Read in csvs as Pandas DataFrames
    filepath_true = os.path.join(dir, "data2_true.csv")
    filepath_fake = os.path.join(dir, "data2_fake.csv")
    true_df = pd.read_csv(filepath_true, usecols=["subject", "text"])
    fake_df = pd.read_csv(filepath_fake, usecols=["subject", "text"])

    # Filter out all rows which are not political news
    true_df = true_df[true_df["subject"] == "politicsNews"]
    fake_df = fake_df[fake_df["subject"] == "politics"]

    # Rename the "subject" column to "label"
    true_df.rename(columns={"text": "Document", "subject": "Label"}, inplace=True)
    fake_df.rename(columns={"text": "Document", "subject": "Label"}, inplace=True)

    # Change all entries in "label" column to their corresponding label 0: FakeNews; 1: RealNews
    true_df["Label"] = 1
    fake_df["Label"] = 0

    # Combine both DataFrames
    df = pd.concat([true_df, fake_df], axis=0)

    # Remove whitespace rows in the DataFrame
    df = df[df["Document"].str.strip() != ''].reset_index(drop=True)

    # Swap the column position
    df = df[['Label', 'Document']]
    
    # Mark the data as belonging to dataset 2
    df["Dataset"] = 2

    return df


def get_data3_df(dir):
    """
    Read in LIAR dataset as a DataFrame
    - Download URL: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
    - Train, test, and validation set combined to form a single dataset
    - Only consider labels of "true", "mostly-true", "false", and "pants-fire"

    :param dir: Path to directory for raw data
    :return:    DataFrame of LIAR dataset
    """
    
    # Read in tsvs as Pandas DataFrames and combine them into a single DataFrame
    filepath_test = os.path.join(dir, "data3_test.tsv")
    filepath_train = os.path.join(dir, "data3_train.tsv")
    filepath_val = os.path.join(dir, "data3_val.tsv")
    test_df = pd.read_csv(filepath_test, sep="\t", header=None, usecols=[1, 2])
    train_df = pd.read_csv(filepath_train, sep="\t", header=None, usecols=[1, 2])
    val_df = pd.read_csv(filepath_val, sep="\t", header=None, usecols=[1, 2])
    df = pd.concat([test_df, train_df, val_df], axis=0).reset_index(drop=True)
    
    # Rename the column headers
    df = df.rename(columns={df.columns[0]: "Label", df.columns[1]: "Document"})
    
    # Only consider statements which are most likely real or fake
    allowed_labels = ["true", "mostly-true", "false", "pants-fire"]
    df = df[df["Label"].isin(allowed_labels)].reset_index(drop=True)
    
    # Change all entries in "label" column to their corresponding label 0: FakeNews; 1: RealNews
    pd.set_option('future.no_silent_downcasting', True)
    label_mapping = {"true": 1, "mostly-true": 1, "false": 0, "pants-fire": 0}
    df["Label"] = df["Label"].replace(label_mapping)

    # Mark the data as belonging to dataset 3
    df["Dataset"] = 3

    return df


def get_statistics(df):
    """
    Compute statistics for a DataFrame, which includes:
    (1) Number of Datapoints
    (2) Distribution of Datapoints
    (3) Number of Words (min, max, avg)
    (4) Number of Sentences (min, max, avg)

    :param df: DataFrame to extract statistics from
    :return:   String containing statistics
    """

    def count_words(text):
        tokens = word_tokenize(text)
        return len(tokens)

    def count_sentences(text):
        sentences = sent_tokenize(text)
        return len(sentences)

    df["Num_Words"] = df["Document"].apply(count_words)
    df["Num_Sentences"] = df["Document"].apply(count_sentences)

    # Calculate statistics    
    label_distribution = df["Label"].value_counts().to_string()
    num_datapoints = df.shape[0]
    
    min_words = df["Num_Words"].min()
    max_words = df["Num_Words"].max()
    avg_words = df["Num_Words"].mean()

    min_sentences = df["Num_Sentences"].min()
    max_sentences = df["Num_Sentences"].max()
    avg_sentences = df["Num_Sentences"].mean()

    # Add statistics to string
    statistics_str = (f"{label_distribution}\n"
                      f"Total: {num_datapoints}\n"
                      f"Words - Min: {min_words} Max: {max_words} Avg: {avg_words:.1f}\n"
                      f"Sentences - Min: {min_sentences} Max: {max_sentences} Avg: {avg_sentences:.1f}\n\n")

    return statistics_str


def write_data(train_df, test_df, stats):
    """
    Write DataFrame and string objects to CSV and text files respectively
    - Statistics will be written to "statistics.txt"
    - Training data will be written to "train_df"
    - Testing data will be written to "test_df"

    :param train_df: DataFrame containing the training data
    :param test_df:  DataFrame containing the test data
    :param stats:    String containing statistical data
    """
    
    # Define the directory to save output files
    out_dir = os.path.join(os.getcwd(), "dataset", "combine_data")

    # Save statistics into text file
    stats_filepath = os.path.join(out_dir, "statistics.txt")
    with open(stats_filepath, "w") as stats_file:
        stats_file.write(stats)

    # Save training DataFrame into CSV file
    df_filepath = os.path.join(out_dir, "train.csv")
    train_df.to_csv(df_filepath, index=False)

    # Save testing DataFrame into CSV file
    df_filepath = os.path.join(out_dir, "test.csv")
    test_df.to_csv(df_filepath, index=False)


if __name__ == "__main__":
    train_df, test_df, stats = combine()
    write_data(train_df, test_df, stats)
