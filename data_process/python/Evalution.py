import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils2.dataset import Track_dataset
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
from scipy.linalg import sqrtm

# warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
# from discriminative_metrics import discriminative_score_metrics
# from predictive_metrics import predictive_score_metrics
# from visualization_metrics import visualization

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
        
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  # 双向，所以 *2
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize the hidden state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        if not self.use_gru:
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
            out, _ = self.rnn(x, (h0, c0))  # LSTM
        else:
            out, _ = self.rnn(x, h0)  # GRU
        
        # Take the hidden state at the last time step as the feature
        out = out[:, -1, :]
        out = self.fc(out)  # (batch_size, hidden_dim)
        return out

##################################################FID CACULATE#############################################################
# Compute feature vectors using the feature extractor
def calculate_feature_vectors(images, model):
    model.eval()
    with torch.no_grad():
        features = model(images)
    return features.cpu().numpy()

# Compute mean and covariance
def calculate_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

# 计算FID
def calculate_fid(real_features, fake_features):


    mu_real, sigma_real = calculate_statistics(real_features)
    mu_fake, sigma_fake = calculate_statistics(fake_features)

    sigma_real += np.eye(sigma_real.shape[0]) * 1e-6
    sigma_fake += np.eye(sigma_fake.shape[0]) * 1e-6


    # Compute the Frobenius norm between feature vectors
    diff = mu_real - mu_fake
    cov_mean_sqrt = sqrtm(sigma_real.dot(sigma_fake))
    if np.iscomplexobj(cov_mean_sqrt):
        cov_mean_sqrt = cov_mean_sqrt.real
    fid = np.sum(diff**2) + np.trace(sigma_real + sigma_fake - 2 * cov_mean_sqrt)
    return fid

# ------------------------------- load classifier ---------------------------------
def fid_score(classifier, real_dataset, fake_dataset, batch_size = 128): # input should be torch dataset

    # Load real and generated data
    real_data_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True)
    fake_data_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=True)



    # Extract feature vectors from real and generated data
    real_features = []
    fake_features = []

    for x, _ in real_data_loader:

        real_features.append(calculate_feature_vectors(x.to('cuda'), classifier))

    for x, _ in fake_data_loader:
        x = x.squeeze(1)
        # noise = torch.randn(x.shape)

        fake_features.append(calculate_feature_vectors(x.to('cuda'), classifier))

    # Concatenate feature vectors into a single NumPy array
    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)

    # calcu FID

    fid_score = calculate_fid(real_features, fake_features)
    return fid_score
#######################################################################################################


class ModifiedDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]

        # Reshape x from (C, H, W) to (H*W, C)

        x = x.view(x.shape[0]*x.shape[1], x.shape[2])  # 转换为 (H*W, C)
        x = x.permute(1,0)
        # x = torch.where(x == -1, torch.tensor(0.0), x)

        return x, y


#####################################################################################################
def gaussian_kernel(x, y, sigma=1.0):
    """
    Compute Gaussian kernel.
    x: (m, d), y: (n, d)
    Returns: kernel matrix of shape (m, n)
    """
    # Expand dimensions to compute pairwise Euclidean distances

    diff = x.unsqueeze(1) - y.unsqueeze(0)  # (m, n, d)
    dist_sq = torch.sum(diff ** 2, dim=2)     # (m, n)
    kernel = torch.exp(-dist_sq / (2 * sigma ** 2))
    return kernel

def compute_mmd(x, y, sigma=1.0):
    """
    Compute MMD^2. The inputs x and y are feature vectors extracted 
    from real and generated data, with shapes (m, d) and (n, d), respectively.
    """

    m = x.size(0)
    n = y.size(0)
    K_xx = gaussian_kernel(x, x, sigma)
    K_yy = gaussian_kernel(y, y, sigma)
    K_xy = gaussian_kernel(x, y, sigma)
    
    mmd = K_xx.sum()/(m*m) + K_yy.sum()/(n*n) - 2*K_xy.sum()/(m*n)
    return 1000 * mmd

def polynomial_kernel(x, y, degree=3, coef0=1):
    """
    Polynomial kernel computation: k(x, y) = ((x^T y)/d + coef0)^degree,
    where d is the feature dimension.
    """

    d = x.size(1)
    return (torch.matmul(x, y.t()) / d + coef0) ** degree

def compute_kid(x, y, degree=3, coef0=1):
    """
    Compute the unbiased estimate of KID (Kernel Inception Distance).
    x and y are the features extracted from real and generated data, 
    with shapes (m, d) and (n, d), respectively.
    """
    m = x.size(0)
    n = y.size(0)
    K_xx = polynomial_kernel(x, x, degree, coef0)
    K_yy = polynomial_kernel(y, y, degree, coef0)
    K_xy = polynomial_kernel(x, y, degree, coef0)
    

    sum_K_xx = (K_xx.sum() - torch.diag(K_xx).sum()) / (m * (m - 1))
    sum_K_yy = (K_yy.sum() - torch.diag(K_yy).sum()) / (n * (n - 1))
    sum_K_xy = K_xy.mean()
    
    kid = sum_K_xx + sum_K_yy - 2 * sum_K_xy
    return 1000 * kid

def calculate_mmdandkid(classifier, fake_dataset, real_dataset, batch_size=128):

    real_data_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True)
    fake_data_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=True)


    real_features = []
    fake_features = []

    for x, _ in real_data_loader:
        real_features.append(calculate_feature_vectors(x.to('cuda'), classifier))

    for x, _ in fake_data_loader:
        x = x.squeeze(1)
        fake_features.append(calculate_feature_vectors(x.to('cuda'), classifier))

    real_feats = torch.cat([torch.from_numpy(f) for f in real_features], dim=0)
    fake_feats = torch.cat([torch.from_numpy(f) for f in fake_features], dim=0)


    sigma = 1.0  #
    mmd_value = compute_mmd(real_feats, fake_feats, sigma)
    kid_value = compute_kid(real_feats, fake_feats, degree=3, coef0=1)

    return mmd_value.item(), kid_value.item()

########################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# np.random.seed(123)
real_dataset = torch.load('./processed_data/Track_dataset_smooth.pth')
real_dataset_list = []

for i in range(5):
    indices = np.random.choice(len(real_dataset), 256, replace=False)
    real_subset = Subset(real_dataset, indices)
    real_dataset_list.append(real_subset)



fake_data = np.load('./generated_data/track_data.npy')
fake_data = fake_data.reshape(fake_data.shape[0], 3, 12, 140)

fake_label = [(0, 0) for _ in range(len(fake_data))]
fake_dataset = Track_dataset(data=fake_data, labels=fake_label)



model = RNNFeatureExtractor()
model.load_state_dict(torch.load('./Contrastive_Learning_model/rnn_feature_extractor.pth'))
model = model.to(device)

fake_dataset = ModifiedDataset(fake_dataset)

fid_list = []
for i in range(5):
    real_dataset = ModifiedDataset(real_dataset_list[i])


    fid = fid_score(model, fake_dataset, real_dataset)
    fid_list.append(fid)
print(np.mean(fid_list))


mmd, kid = calculate_mmdandkid(model, fake_dataset, real_dataset)
print(mmd, kid)