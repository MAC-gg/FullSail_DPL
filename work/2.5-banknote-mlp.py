
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F


# Use the MLP class from the notebooks
class SimpleMLP(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=16, output_dim=1):
        ''' Define the shape of the network
        '''
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        '''Define how data flows through network
        '''
        x = self.hidden(x)
        x = F.relu(x)
        
        x = self.output(x)
        x = torch.sigmoid(x)
        
        return x


if __name__ == '__main__':
    # Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '../data/banknote-auth.csv')
    df = pd.read_csv(csv_path, encoding='latin-1')

    # Apply a train / test split
    X = df.drop('label', axis=1)
    y = df['label']
    torch.manual_seed(51)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=51)

    # Convert to torch tensors
    X_train = torch.from_numpy(X_train.values.copy()).float()
    X_test = torch.from_numpy(X_test.values.copy()).float()
    y_train = torch.from_numpy(y_train.values.copy()).float().view(-1, 1)
    y_test = torch.from_numpy(y_test.values.copy()).float().view(-1, 1)

    # Create a small MLP with at least one hidden layer
    model = SimpleMLP(input_dim=4, hidden_dim=16, output_dim=1)

    # Train for a reasonable number of epochs (e.g., 10–20)
    # Use the standard loop: forward > loss > zero_grad > backward > step
    loss_fn = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 50
    for _ in range(epochs):
        pred = model(X_train)
        loss = loss_fn(pred, y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_preds = (model(X_train) > 0.5).float()
        test_preds = (model(X_test) > 0.5).float()

        train_acc = accuracy_score(y_train.numpy(), train_preds.numpy())
        test_acc = accuracy_score(y_test.numpy(), test_preds.numpy())

        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        # Create a confusion matrix to summarize results
        cm = confusion_matrix(y_test.numpy(), test_preds.numpy())
        print("Confusion Matrix:\n", cm)


# Reflection


# 1 Did the model perform similarly on the training and test sets?
# ======= Yes, only about a single point off. 


# 2 What happens if we modify the learning and epochs?
# ======= When the learning rate is increased to 0.01, the model's accuracy improves to 0.98.
# The higher the learning rate, the less epochs are needed to reach a good accuracy.

# ======= When the epochs were increased to 50, the accuracy decreased by about 10 points.
# Then I set some random states to see if that was the issue, and the accuracy only dropped about 5 points.
# If we are decreasing with more epochs, it is likely that the model is overfitting to the training data and not generalizing well to the test data.

# ======= When both epochs and learning rate were increased (epochs=50, lr=0.01), the accuracy was nearly perfect.
# This is likely due to the model having more training opprotunities and a higher learning rate allowing it to converge faster.


# 3 How happens if you adjust the size of the hidden layer?
# ======= When the hidden dim was decreased to 8, there was a small dropoff in accuracy (about 0.7) and a slight underfit meaning the model needs to be more layers to better predict the data.
# When the hidden dim was increased to 32, there was a giant dropoff in accuracy (about 0.48).
# A coincidence here is that when decreased the model only missed in one direction, but when increased, the model missed in only the opposite direction.

# 8 hidden dim:
# Confusion Matrix:
# [[194   0]
# [ 99  50]]

# 32 hidden dim:
# Confusion Matrix:
# [[ 13 181]
# [  0 149]]
