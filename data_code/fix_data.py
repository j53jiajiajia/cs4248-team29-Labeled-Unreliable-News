"""
Author: Daren Tan (Team 29)
Date: Mar 22, 2024
Description: Python script to split train set output from combine_data.py into train-test set

1) This code should be executed after combine_data.py, since it's meant to fix dataset generated from that
2) Harin discovered that data1_test.csv was actually a subset of data1_fulltrain.csv. 
3) To prevent data leakage during model training and testing, our group decided to use data1_fulltrain.csv 
   to form our train-test set
4) Duplicates within data1_fulltrain.csv were also removed
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the training data into Pandas DataFrame
file_path = os.path.join(os.getcwd(), "dataset", "combine_data", "train.csv")
df = pd.read_csv(file_path)

# Remove duplicate copies in dataset
df = df.drop_duplicates(subset=["Document"], keep='first')

# Split into training and testing data, stratified over Label and Dataset
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[["Label", "Dataset"]], random_state=4248)

# Reset the indexes of the DataFrames
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Save DataFrames into CSV files
out_filepath = os.path.join(os.getcwd(), "dataset", "fixed_data_1")
train_df.to_csv(os.path.join(out_filepath, "train.csv"), index=False)
test_df.to_csv(os.path.join(out_filepath, "test.csv"), index=False)

# Print dataframe to check output
print(train_df)
print(test_df)