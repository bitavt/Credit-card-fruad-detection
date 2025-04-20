import torch
import torch.nn as nn
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,roc_auc_score,
    precision_score, recall_score, confusion_matrix, f1_score
)
import umap
import plotly.express as px

# custom imports
from constants import *



class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            # nn.Linear(4, 2),
            # nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            # nn.Linear(2, 4),
            # nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        # latent
        encoded = self.encoder(x)
        # reconstructed
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoderAnomalyDetection:
    def __init__(self,
                 train_loader,
                 val_loader,
                 test_loader,
                 device
                 ):
        self.train_loader= train_loader
        self.val_loader= val_loader
        self.test_loader= test_loader
        self.epoch= EPOCHS
        self.device= device
        # initialize auto encoder
        self.model= AutoEncoder(
            input_dim= len(self.train_loader.dataset[0][0])
        ).to(device)
        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        # Initialize the loss function
        self.loss_fn = nn.MSELoss()

    def training(self):
        trainingEpoch_loss = []
        valEpoch_loss = []

        # def train(train_loader,test_loader,optimizer,loss_fn,model,num_epochs):
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}\n-------------------------------")
            #-------------------
            # train
            #-------------------
            train_size = len(self.train_loader.dataset)
            self.model.train()
            train_batch_loss= []
            for batch, (X,y) in enumerate(self.train_loader):
                X, y = X.to(self.device), y.to(self.device)
                # compute prediction and loss
                X_reconstructed = self.model(X)
                loss = self.loss_fn(X_reconstructed, X)
                # Zero gradients, perform a backward pass, and update the weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # batch loss
                train_batch_loss.append(loss.item())
                if batch % 1000 == 0:
                    loss, current = loss.item(), batch * BATCH_SIZE + len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{train_size:>5d}]")
            trainingEpoch_loss.append(np.array(train_batch_loss).mean())

            #-----------------
            # validate
            #-----------------
            self.model.eval()
            # size = len(dataloader.dataset)
            num_batches = len(self.val_loader)
            val_loss= 0

            with torch.no_grad():
                val_batch_loss= []
                for X, y in self.val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    # compute prediction and loss
                    X_reconstructed = self.model(X)
                    loss_val = self.loss_fn(X_reconstructed, X).item()
                    val_loss += loss_val
                    val_batch_loss.append(loss_val)
            valEpoch_loss.append(np.array(val_batch_loss).mean())
            val_loss /= num_batches
            print(f"Val Error: \n Avg loss: {val_loss:>8f} \n")

        # plot the training process
        plt.plot(trainingEpoch_loss, label='training loss')
        plt.plot(valEpoch_loss, label='validation loss')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.show()

    def form_reconstruction_error(self):
        # anomaly detection
        self.model.eval()
        reconstruction_errors = []
        actual_labels = []
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                X_reconstructed = self.model(X)
                test_loss = torch.mean((X_reconstructed - X) ** 2, dim=1)
                reconstruction_errors.extend(test_loss.cpu().numpy())
                actual_labels.extend(y.cpu().numpy())

        # Convert the lists into NumPy arrays
        self.reconstruction_errors= np.array(reconstruction_errors)
        self.actual_labels= np.array(actual_labels)


    def evaluate_model(self, threshold):
        print(f"\nEvaluating model with threshold = {threshold:.4f}...")
        print("==============================================================")
        # make a prediction
        predictions = (self.reconstruction_errors > threshold).astype(int)

        # classification report
        print("\nClassification Report:")
        print(classification_report(self.actual_labels, predictions, target_names=["Normal", "Fraud"]))

        # roc auc
        roc_auc = roc_auc_score(self.actual_labels, predictions)
        print(f"ROC AUC score: {roc_auc:.4f}")

        # plot confusion matrix
        cm = confusion_matrix(self.actual_labels, predictions)
        sns.heatmap(cm, linewidths=.12, cmap="coolwarm", annot=True, fmt=".1f")
        plt.title("Confusion matrix")
        plt.show()

        # plot reconstruction error distribution
        self.plot_reconstruction_error_dist(thresholds=threshold)

        return {
            "threshold": threshold,
            "roc_auc": roc_auc,
            "f1": f1_score(self.actual_labels, predictions),
            # "precision": np.nan_to_num(cm[1,1] / (cm[0,1] + cm[1,1])),  # TP / (FP + TP)
            # "recall": np.nan_to_num(cm[1,1] / (cm[1,0] + cm[1,1]))     # TP / (FN + TP)
            "precision": precision_score(self.actual_labels, predictions),  # TP / (FP + TP)
            "recall": recall_score(self.actual_labels, predictions)     # TP / (FN + TP)
        }

    def plot_reconstruction_error_dist(self, thresholds=None):
        """
        Plot the reconstruction error distribution for normal and fraud classes,
        and optionally mark thresholds on the plot.

        Args:
            thresholds (float or list of floats): Threshold(s) to draw as vertical lines.
        """
        plt.figure(figsize=(8, 6))
        sns.histplot(self.reconstruction_errors[self.actual_labels == 0], bins=50, label="Normal", color='blue', stat='density', kde=True)
        sns.histplot(self.reconstruction_errors[self.actual_labels == 1], bins=50, label="Fraud", color='red', stat='density', kde=True)

        # Handle single threshold or list of thresholds
        if thresholds is not None:
            if not isinstance(thresholds, (list, tuple, np.ndarray)):
                thresholds = [thresholds]
            for t in thresholds:
                plt.axvline(t, color='black', linestyle='--', label=f"Threshold = {t:.4f}")

        plt.title("Reconstruction Error Distribution")
        plt.xlabel("Reconstruction Error")
        plt.ylabel("Density")
        plt.legend()
        plt.grid()
        plt.show()

    def threshold_sweep(self, thresholds):
        results = []

        for t in thresholds:
            result = self.evaluate_model(t)
            results.append(result)

        # Convert to DataFrame for easy plotting
        df = pd.DataFrame(results)

        # Plot F1 and ROC-AUC over thresholds
        plt.figure(figsize=(8, 6))
        plt.plot(df["threshold"], df["f1"], label="F1 Score")
        plt.plot(df["threshold"], df["roc_auc"], label="ROC AUC")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Performance Metrics vs. Threshold")
        plt.legend()
        plt.grid()
        plt.show()

        return df


    def visualize_latent_space(self, mode='3d'):
        """
        Visualize latent space using Plotly in 2D or 3D interactively.

        Args:
            mode (str): '2d' or '3d'
        """
        assert mode in ['2d', '3d'], "Mode must be either '2d' or '3d'"

        latent_reps = []
        actual_labels = []

        self.model.eval()
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                encoded = self.model.encoder(X)
                latent_reps.append(encoded.cpu().numpy())
                actual_labels.extend(y.cpu().numpy())

        latent_reps = np.concatenate(latent_reps)
        actual_labels = np.array(actual_labels)

        reducer = umap.UMAP(n_components=3 if mode == '3d' else 2, random_state=42)
        latent_umap = reducer.fit_transform(latent_reps)

        df = pd.DataFrame(latent_umap, columns=["Dim1", "Dim2", "Dim3"][:latent_umap.shape[1]])
        df["label"] = ["Fraud" if l == 1 else "Non-fraud" for l in actual_labels]

        if mode == '3d':
            fig = px.scatter_3d(
                df, x="Dim1", y="Dim2", z="Dim3",
                color="label",
                symbol="label",
                opacity=0.8,
                title="3D Latent Space Visualization",
                color_discrete_map={"Non-fraud": "blue", "Fraud": "red"}
            )
        else:
            fig = px.scatter(
                df, x="Dim1", y="Dim2",
                color="label",
                symbol="label",
                opacity=0.8,
                title="2D Latent Space Visualization",
                color_discrete_map={"Non-fraud": "blue", "Fraud": "red"}
            )

        fig.update_layout(legend_title_text='Class')
        fig.show()

    # def visualize_latent_space(self):
    #     # Get the latent (encoded) representations
    #     latent_reps = []
    #     actual_labels = []

    #     self.model.eval()
    #     with torch.no_grad():
    #         for X,y in self.test_loader:
    #             X, y = X.to(self.device), y.to(self.device)
    #             # find the latent representation
    #             encoded = self.model.encoder(X)
    #             latent_reps.append(encoded.cpu().numpy())
    #             actual_labels.extend(y.cpu().numpy())

    #     latent_reps = np.concatenate(latent_reps)
    #     actual_labels = np.array(actual_labels)

    #     reducer= umap.UMAP(n_components=2, random_state=42)
    #     latent_umap= reducer.fit_transform(latent_reps)

    #     # Plot the UMAP projection:
    #     plt.figure(figsize=(8, 6))
    #     # Different markers or colors for fraud vs. non-fraud
    #     for label, marker, color in zip([0, 1], ['o', 'x'], ['blue', 'red']):
    #         indices = np.where(actual_labels == label)
    #         plt.scatter(latent_umap[indices, 0], latent_umap[indices, 1],
    #                     marker=marker, color=color,
    #                     label='Non-fraud' if label == 0 else 'Fraud', alpha=0.7, s=50)

    #     plt.xlabel("UMAP Dimension 1")
    #     plt.ylabel("UMAP Dimension 2")
    #     plt.title("UMAP Projection of the Latent Space")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()