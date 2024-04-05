"""
Author: Harin (Team 29)
Date: Mar 28, 2024
Description: Python script to train DistilBERT model on classification

Original Colab Notebook: https://drive.google.com/file/d/1LOLEhvzjly3vI08loNUVtrzC8XzwxdXn/view?usp=drive_link
"""

import os
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, DistilBertModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NUM = 4
MAX_LENGTH = 200

class DistilBERTClassifier(nn.Module):
    
    def __init__(self):
        super(DistilBERTClassifier, self).__init__()
        self.distil_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 8)
        self.out = nn.Linear(8, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        # BERT Model
        bert_output = self.distil_bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = bert_output[0]
        pooler = hidden_state[:, 0]

        # Hidden Layer 1
        x = self.fc1(pooler)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Hidden Layer 2
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Hidden Layer 3
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        # Output Layer
        x = self.out(x)
        log_probs = self.log_softmax(x)
        
        return log_probs


def read_data():
    """
    Read train, validation, and test data from CSV files

    :return: List of data and labels from each of the sets
    """
    # Read data from CSV files
    data_directory = os.path.join(os.path.dirname(os.getcwd()), "data_code", "dataset", "fixed_data_2")
    train_df = pd.read_csv(os.path.join(data_directory, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_directory, "test.csv"))
    val_df = pd.read_csv(os.path.join(data_directory, "val.csv"))

    # Extract Data (X) and Labels (y) from DataFrames
    X_train, y_train = train_df["Document"], train_df["Label"]
    X_test, y_test = test_df["Document"], test_df["Label"]
    X_val, y_val = val_df["Document"], val_df["Label"]

    return [X_train, y_train, X_test, y_test, X_val, y_val]


def tokenize_data(data, max_length):
    """
    Tokenize the data and save them into DataLoaders

    :param data:       List of data and labels from each of the sets
    :param max_length: Controls the token length for padding/truncation
    :return:           DataLoaders for train, validation, and tests datasets
    """

    # Use the AutoTokenizer from HuggingFace library
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", truncation=True, do_lower_case=True, use_fast=True)
    
    # Tokenizing the training set.
    train_tokens = tokenizer.batch_encode_plus(data[0].tolist(), max_length=max_length, padding="max_length", truncation=True)
    val_tokens = tokenizer.batch_encode_plus(data[4].tolist(), max_length=max_length, padding="max_length", truncation=True)
    test_tokens = tokenizer.batch_encode_plus(data[2].tolist(), max_length=max_length, padding="max_length", truncation=True)

    # Storing data into DataLoaders
    train_loader = create_dataloader(train_tokens, data[1])
    val_loader = create_dataloader(val_tokens, data[5])
    test_loader = create_dataloader(test_tokens, data[3])

    return train_loader, val_loader, test_loader


def create_dataloader(X_tokens, y_labels, batch_size=32):
    """
    Create DataLoader given the tokenized data and labels

    :param X_tokens:   Data after being tokenized
    :param y_labels:   Labels of the corresponding data
    :param batch_size: Number of samples processed in one forward and backward pass
    :return:           DataLoader containing data, labels, and batch size
    """
    X_seq = torch.tensor(X_tokens['input_ids'])
    X_mask = torch.tensor(X_tokens['attention_mask'])
    y_labels = torch.tensor(y_labels.tolist())
    dataset = TensorDataset(X_seq, X_mask, y_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def train_and_val(train_loader, val_loader):
    """
    Train and run validation on DistilBERT model over several epochs

    :param train_loader: DataLoader containing training data
    :param val_loader:   DataLoader containing validation data
    """
    model = DistilBERTClassifier().to(device)
    
    # Define parameters to train on
    num_epochs = 3
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Store the training and validation losses
    best_val_loss = float('inf')
    training_losses = []
    validation_losses = []

    # Define and create output directory to store model weights
    output_directory = os.path.join(os.getcwd(), "model_weights")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for epoch in range(num_epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, num_epochs))

        training_loss = train(model, train_loader, optimizer, criterion)
        validation_loss = evaluate(model, val_loader, criterion)
        
        # Save the best performing model
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            output_filepath = os.path.join(output_directory, f"DistilBERT_LUN_weights_{MODEL_NUM}.pt")
            torch.save(model.state_dict(), output_filepath)
        
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        print(f'\nTraining Loss: {training_loss:.3f}')
        print(f'Validation Loss: {validation_loss:.3f}')


def train(model, train_loader, optimizer, criterion):
    """
    Train the model over 1 epoch

    :param model:        Model to be trained (DistilBERTClassifier)
    :param train_loader: DataLoader containing training data
    :param optimizer:    Component responsible for updating parameters
    :param criterion:    Component that defines the loss function
    :return:             Average loss across all batches
    """
    # Set model to training mode
    model.train()
    
    total_loss = 0.0
    with tqdm(total=len(train_loader)) as pbar:

        # Loop over each batch in the data loader
        for batch in train_loader:
            
            # Extract data for each batch and move them to device
            X_batch_seq, X_batch_mask, Y_batch = batch
            X_batch_seq, X_batch_mask, Y_batch = X_batch_seq.to(device), X_batch_mask.to(device), Y_batch.to(device)

            # Calculate loss for each batch and add them to total loss
            log_probs = model(X_batch_seq, X_batch_mask)                
            loss = criterion(log_probs, Y_batch)
            total_loss = total_loss + loss.item()

            # Update model based on the newly computed gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Prevent further computation using gradient of current batch
            log_probs = log_probs.detach().cpu().numpy()

            pbar.update(1)
        
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, val_loader, criterion):
    """
    Evaluate the model on the validation set

    :param model:      Model to be trained (DistilBERTClassifier)
    :param val_loader: DataLoader containing validation data
    :param criterion:  Component that defines the loss function
    """
    # Set model to evaluation mode
    model.eval()

    total_loss = 0.0
    with tqdm(total=len(val_loader)) as pbar:
        
        # Loop over each batch in the data loader
        for batch in val_loader:
            
            # Extract data for each batch and move them to device
            X_batch_seq, X_batch_mask, Y_batch = batch
            X_batch_seq, X_batch_mask, Y_batch = X_batch_seq.to(device), X_batch_mask.to(device), Y_batch.to(device)

            # Disable gradient computation since we are only evaluating the model
            with torch.no_grad():
                
                # Compute validation loss between actual and predicted values and add them to total
                log_probs = model(X_batch_seq, X_batch_mask) 
                loss = criterion(log_probs, Y_batch)
                total_loss = total_loss + loss.item()

                # Prevent further computation using gradient of current batch
                log_probs = log_probs.detach().cpu().numpy()

            pbar.update(1)
        
    # Set model back to train mode.
    model.train()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


def predict_and_eval(test_loader):
    """
    Evaluate model on the training dataset.
    The best model weights derived from the train_and_val function are loaded.

    :param test_loader: DataLoader containing testing data
    """
    # Load pretrained model weights
    model_filepath = os.path.join(os.getcwd(), "model_weights", f"DistilBERT_LUN_weights_{MODEL_NUM}.pt")
    model = DistilBERTClassifier().to(device)
    model.load_state_dict(torch.load(model_filepath))

    # Set model to evaluation mode
    model.eval()
    
    Y_pred, Y_test = [], []
    with tqdm(total=len(test_loader)) as pbar:
        
        # Loop over each batch in the data loader
        for batch in test_loader:
            
            # Extract data for each batch and move them to device
            X_batch_seq, X_batch_mask, Y_batch = batch
            X_batch_seq, X_batch_mask, Y_batch = X_batch_seq.to(device), X_batch_mask.to(device), Y_batch.to(device)

            # Compute the predicted label from highest log probability
            log_probs = model(X_batch_seq, X_batch_mask)                
            Y_batch_pred = torch.argmax(log_probs, dim=1)

            # Append predictions and ground truth for current batch
            Y_test += list(Y_batch.detach().cpu().numpy())
            Y_pred += list(Y_batch_pred.detach().cpu().numpy())

            pbar.update(1)

    # Reset the model back to training mode
    model.train()
    
    # Generate and print classifcation report
    report = classification_report(Y_test, Y_pred, digits=5)
    print(report)

    # Define and create directory to save report if does not exist
    report_directory = os.path.join(os.getcwd(), "report")
    if not os.path.exists(report_directory):
        os.makedirs(report_directory)
    
    # Save classification report as text file
    report_filepath = os.path.join(report_directory, f"report_{MODEL_NUM}.txt")
    with open(report_filepath, 'w') as file:
        file.write(report)
    

if __name__ == '__main__':
    print("Step 1 of 3: Load and tokenize data")
    data = read_data()
    train_loader, val_loader, test_loader = tokenize_data(data, MAX_LENGTH)

    print("Step 2 of 3: Train model")
    train_and_val(train_loader, val_loader)

    print("Step 3 of 3: Test model")
    predict_and_eval(test_loader)
