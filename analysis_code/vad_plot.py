"""
Author: Daren Tan (Team 29)
Date: Apr 17, 2024
Description: Python script to generate 3D cluster plot of data from VAD values
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read in files containing features
feature_directory = "/mnt/store/tjunheng/Project"
train_feature_df = pd.read_csv(os.path.join(feature_directory, f"train_features0.csv"))
test_feature_df = pd.read_csv(os.path.join(feature_directory, f"test_features0.csv"))
full_df = pd.concat([train_feature_df, test_feature_df], axis=0)

# Increase font size of labels
sns.set_context("talk")

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Scatter plot points with different colors based on class labels
colors = {0: "r", 1: "b"}
for label, group in full_df.groupby("Label"):
    ax.scatter(group["valence"], group["arousal"], group["domination"], c=colors[label], label=label)

# Set labels for each axis
ax.set_xlabel("valence")
ax.set_ylabel("arousal")
ax.set_zlabel("domination")

# Add legend
ax.legend()

# Save the image
plt.savefig('clustering_plot.png')