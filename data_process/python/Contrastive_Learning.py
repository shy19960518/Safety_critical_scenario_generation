# Contrastive Learning allows the RNN extractor to learn more discriminative trajectory features, thereby better calculating FID and cosine similarity.
# We use NT-Xent (Normalized Temperature-scaled Cross Entropy Loss), similar to SimCLR, to train RNN.

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
# ============================
# 1. RNN Feature Extractor
# ============================
class RNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim=36, hidden_dim=128, num_layers=2, use_gru=True):
        super(RNNFeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gru = use_gru
        
        if use_gru:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        if not self.use_gru:
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
            out, _ = self.rnn(x, (h0, c0))  # LSTM
        else:
            out, _ = self.rnn(x, h0)  # GRU
        
        # Take the hidden state of the last time step as the feature
        out = out[:, -1, :]
        out = self.fc(out)  # (batch_size, hidden_dim)
        return out

# ============================
# 2. Data Augmentation
# ============================
def augment_trajectory(data, noise_std=0.01):
    """Perturb the trajectory data (x, y, rad) slightly"""
    noise = torch.randn_like(data) * noise_std
    return data + noise

# ============================
# 3. NT-Xent Contrastive loss
# ============================
def contrastive_loss(features, temperature=0.5):
    """NT-Xent Loss，Compute contrast loss based on cosine similarity"""
    batch_size = features.shape[0] // 2  # 2N Two views corresponding to the enhancement
    features = F.normalize(features, dim=1)  # Normalization
    similarity_matrix = torch.matmul(features, features.T)  # Calculate cosine similarity

    # Create a label, two views belong to the same group
    labels = torch.arange(batch_size, device=features.device)
    labels = torch.cat([labels, labels], dim=0)  # Double the batch size

    # calculate NT-Xent Loss
    logits = similarity_matrix / temperature  # Temperature Scaling
    loss = F.cross_entropy(logits, labels)

    return loss

# ============================
# 4. Custom trajectory dataset
# ============================
class ModifiedDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]

        # Preprocessing: Converting Shapes (C, H, W) → (H*W, C)
        x = x.view(x.shape[0] * x.shape[1], x.shape[2]).permute(1, 0)
        # x = torch.where(x == -1, torch.tensor(0.0), x)

        # Generate data augmentation version
        x_aug = augment_trajectory(x)

        return x, x_aug  # 返回 (原始数据, 增强数据)
# ============================
# 5. Training the Contrastive Learning Model
# ============================
def train_rnn_contrastive(model, dataloader, num_epochs=100, lr=0.001, temperature=0.5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x, x_aug in dataloader:
            x = x.to(device)
            x_aug = x_aug.to(device)

            features = model(torch.cat([x, x_aug], dim=0))  # (2N, feature_dim)
            loss = contrastive_loss(features, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    print("Contrastive Learning Training Complete.")
    
    base_dir = Path(__file__).parent.resolve()
    save_path = base_dir / 'deepcluster_model' / 'rnn_feature_extractor.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)  
    torch.save(model.state_dict(), save_path)

# ============================
# 6. Training & Evaluation
# ============================
if __name__ == "__main__":
    # Generate fake data (1000 trajectories, 140 steps each, 36-dimensional features)
    num_samples = 1000
    seq_len = 140
    feature_dim = 36


    base_dir = Path(__file__).parent.resolve()
    load_path = base_dir / "processed_data" / "Track_dataset_smooth.pth"
    dataset = torch.load(load_path)


    load_path = (Path(__file__).parent / "processed_data" / "Track_dataset_smooth.pth").resolve()
    dataset = torch.load(load_path)

    # indices = np.random.choice(len(dataset), 512, replace=False)  # Randomly sample N_sample indexes
    # dataset = Subset(dataset, indices)

    dataset = ModifiedDataset(dataset)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    # for x,y in dataloader:
    #     print(x.shape) # (b,l,c)
    #     assert 1==2
    feature_extractor = RNNFeatureExtractor(input_dim=feature_dim, hidden_dim=128, use_gru=True)
    train_rnn_contrastive(feature_extractor, dataloader, num_epochs=100, lr=1e-3, temperature=0.5)
