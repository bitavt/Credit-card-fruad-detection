import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import (
    classification_report,roc_auc_score,
    precision_score, recall_score, confusion_matrix, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import plotly.express as px
import copy

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
    def __init__(
        self,
        train_loader,
        val_loader,
        test_loader,
        device,
        patience: int = 3  # early stopping patience
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.patience = patience

        # initialize autoencoder
        self.model = AutoEncoder(
            input_dim=len(self.train_loader.dataset[0][0])
        ).to(device)

        # optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

    def training(self, max_epochs: int = EPOCHS):
        training_epoch_loss = []
        val_epoch_loss = []

        best_val_loss = np.inf
        best_model_wts = copy.deepcopy(self.model.state_dict())
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            print(f"Epoch {epoch+1}\n{'-'*30}")
            # ---------- Training --------------------------------------
            self.model.train()
            train_batch_losses = []
            for batch_idx, (X, y) in enumerate(self.train_loader):
                X = X.to(self.device)
                # forward + backward + optimize
                self.optimizer.zero_grad()
                X_recon = self.model(X)
                loss = self.loss_fn(X_recon, X)
                loss.backward()
                self.optimizer.step()

                train_batch_losses.append(loss.item())
                if batch_idx % 1000 == 0:
                    current = batch_idx * BATCH_SIZE + len(X)
                    print(f"  [train] loss: {loss.item():.6f}  [{current}/{len(self.train_loader.dataset)}]")

            avg_train_loss = np.mean(train_batch_losses)
            training_epoch_loss.append(avg_train_loss)

            # ---------- Validation -------------------------------------
            self.model.eval()
            val_batch_losses = []
            with torch.no_grad():
                for X, y in self.val_loader:
                    X = X.to(self.device)
                    X_recon = self.model(X)
                    loss_val = self.loss_fn(X_recon, X).item()
                    val_batch_losses.append(loss_val)

            avg_val_loss = np.mean(val_batch_losses)
            val_epoch_loss.append(avg_val_loss)
            print(f"  [val]   Avg loss: {avg_val_loss:.6f}\n")

            # ---------- Early Stopping Check ----------------------------
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
                print(f"  Validation loss improved. Resetting patience counter.\n")
            else:
                epochs_no_improve += 1
                print(f"  No improvement in validation loss for {epochs_no_improve} epoch(s).\n")

            if epochs_no_improve >= self.patience:
                print(f"Early stopping triggered! No improvement for {self.patience} epochs.")
                print(f"Training stopped at epoch {epoch+1}.")
                break

        # load best model weights (the one with the lowest validation loss)
        self.model.load_state_dict(best_model_wts)

        # plot losses to visualize training and validation performance
        plt.plot(training_epoch_loss, label='training loss')
        plt.plot(val_epoch_loss, label='validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def form_reconstruction_error(self):
        # anomaly detection
        self.model.eval()
        # a list containing reconstruction error on the test set
        reconstruction_errors = []
        # a list containing the actual labels for the test set
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
        print(f"\nEvaluating the model on the test set with threshold = {threshold:.4f}...")
        print("==============================================================")
        # make a prediction (creates a list of binary values based on the threshold)
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
        print("Plotting the reconstruction error distribution...")
        self.plot_reconstruction_error_dist(thresholds=threshold)

        return {
            "threshold": threshold,
            "roc_auc": roc_auc,
            "f1": f1_score(self.actual_labels, predictions),
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
        plt.yscale("log")
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
