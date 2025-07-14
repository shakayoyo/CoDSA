import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import random
import math

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



device ="cuda"

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        """
        t: [batch, 1] tensor with timesteps
        Returns:
            time embedding of shape [batch, embed_dim]
        """
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t * emb  # broadcasting: [batch, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.embed_dim % 2 == 1:  # pad if odd
            emb = F.pad(emb, (0,1))
        return emb

class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features, eps=1e-4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.fc(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, 
                 text_dim, 
                 cond_dim, 
                 hidden_dim=512, 
                 time_embed_dim=128, 
                 num_fc_blocks=5, 
                 dropout=0.1,
                 use_cross_attention=False,
                 num_heads=8):
        """
        Args:
            text_dim (int): Dimensionality of the text embedding.
            cond_dim (int): Dimensionality of the condition (e.g., image) embedding.
            hidden_dim (int): Hidden layer size.
            time_embed_dim (int): Dimensionality of the time embedding.
            num_fc_blocks (int): Number of fully connected blocks.
            dropout (float): Dropout rate.
            use_cross_attention (bool): Whether to add a cross-attention layer.
            num_heads (int): Number of attention heads for cross-attention.
        """
        super().__init__()
        
        # Use sinusoidal time embedding instead of a simple MLP.
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        
        # Input layer: project concatenated (text, cond, time) into hidden_dim.
        self.input_fc = nn.Linear(text_dim + cond_dim + time_embed_dim, hidden_dim)
        
        # Fully connected blocks with dropout.
        self.fc_blocks = nn.ModuleList([
            FCBlock(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_fc_blocks)
        ])
        
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                     num_heads=num_heads,
                                                     dropout=dropout,
                                                     batch_first=True)
            self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        
        # Output layer: project from hidden_dim back to text embedding space.
        self.out_fc = nn.Linear(hidden_dim, text_dim)
        
    def forward(self, x, cond, t):
        """
        Args:
            x: [batch, text_dim] – the current noisy text embedding.
            cond: [batch, cond_dim] – the conditioning embedding.
            t: [batch] – current diffusion timestep (as integers).
        Returns:
            out: [batch, text_dim] – the predicted noise (or denoised embedding).
        """
        t = t.float().unsqueeze(-1)  # shape: [batch, 1]
        t_emb = self.time_embed(t)   # shape: [batch, time_embed_dim]
        
        # Concatenate text, condition, and time embeddings.
        x_in = torch.cat([x, cond, t_emb], dim=-1)
        h = self.input_fc(x_in)
        
        for block in self.fc_blocks:
            h = block(h)
        
        if self.use_cross_attention:
            cond_proj = self.cond_proj(cond)
            query = h.unsqueeze(1)
            key = cond_proj.unsqueeze(1)
            value = key
            attn_out, _ = self.cross_attn(query=query, key=key, value=value)
            h = h + attn_out.squeeze(1)
        
        out = self.out_fc(h)
        return out
    
class DDIMSampler:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n,condition, yshape):
        #logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, yshape), device=self.device)
            for t in reversed(range(1, self.noise_steps)):
                t_tensor = torch.full((n, ), t, device=self.device, dtype=torch.float32)
                predicted_noise = model(x, condition, t_tensor)
                alpha = self.alpha[t]
                alpha_hat = self.alpha_hat[t]
                beta = self.beta[t]
                if t > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        return x
    

def combine_XY(X, y):
    """
    Concatenate X and y .
    """
    return np.hstack([X, y])

# -------------------------------
# Generate Imbalanced Data
# -------------------------------

def generate_imbalanced_data(n_minority, n_majority, beta_coef=None, seed=None):
    """
    Generate data with 5-dimensional features and a nonlinear response.
          
    The region label is defined as:
      region = 1 for minority (u1 < 0.5), 0 for majority (u1 >= 0.5).
    """
    if seed is not None:
        np.random.seed(seed)
    beta_coef = np.array([3.0, 2.0, -1.0, 0.5, 1.0])
    # Generate minority group: u1 in [0, 0.5]
    u1_min = np.random.uniform(0, 0.5, n_minority)
    u2_min = u1_min**2 + np.random.randn(n_minority)
    u3_min = 0.2*np.random.randn(n_minority)
    
    x1_min = u1_min
    x2_min = 0 * u1_min + u2_min  # since I(u1>0.5)=0
    x3_min = 0 * u2_min + np.log(1 + np.abs(u2_min))
    x4_min = np.abs(u2_min)
    x5_min = u1_min - u2_min
    
    X_min = np.column_stack([x1_min, x2_min, x3_min, x4_min, x5_min])
    
    # Nonlinear function for y (minority)
    y_min_clean = X_min.dot(beta_coef.reshape(-1,1))
    additional_min = u3_min.reshape(-1, 1)
    y_min = 2*y_min_clean**2+ y_min_clean + additional_min
    
    # Generate majority group: u1 in [0.5, 1]
    u1_maj = np.random.uniform(0.5, 1, n_majority)
    u2_maj = u1_maj**2 + np.random.randn(n_majority)
    u3_maj = 0.2*np.random.randn(n_majority)
    
    x1_maj = u1_maj
    x2_maj = 1 * u1_maj + u2_maj
    x3_maj = 1 * u2_maj + np.log(1 + np.abs(u2_maj))
    x4_maj = np.abs(u2_maj)
    x5_maj = u1_maj - u2_maj
    
    X_maj = np.column_stack([x1_maj, x2_maj, x3_maj, x4_maj, x5_maj])
    
    y_maj_clean = X_maj.dot(beta_coef.reshape(-1,1))
    additional_maj = u3_maj.reshape(-1, 1)
    y_maj = 2*y_maj_clean**2-y_maj_clean + additional_maj
    
    # Combine groups
    X = np.vstack([X_min, X_maj])
    y = np.vstack([y_min, y_maj])
    
    # Region label: 1 for minority (u1<0.5), 0 for majority.
    regions = np.concatenate([np.zeros(n_minority), np.ones(n_majority)])
    
    return X, y, regions


# -------------------------------
# Synthetic Data Generation for Minority Region
# -------------------------------
def generate_synthetic_data(regions, model, decoder, sampler, text_dim, num_synthetic=500, seed=111):
    set_seed(seed)
    """
    Generate synthetic data for the target (minority) region by oversampling with Gaussian perturbations.
    """
    test_dataloader = DataLoader(regions, batch_size=512, shuffle=False)
    generated_result=[]
    model.eval()
    if decoder == None:
        with torch.no_grad():
            for i in test_dataloader:
                i = i.to(device)
                a=sampler.sample(model,i.shape[0],i,text_dim).cpu().numpy().tolist()
                generated_result=generated_result+ a 
    else:
        with torch.no_grad():
            for i in test_dataloader:
                i = i.to(device)
                a=decoder(sampler.sample(model,i.shape[0],i,text_dim)).cpu().numpy().tolist()
                generated_result=generated_result+ a 
    generated_result = np.array(generated_result)
    return generated_result[:,:-1],  generated_result[:,-1:]

# -------------------------------
# Balanced Split for Training and Validation
# -------------------------------
def balanced_split(X, y, regions, test_size=0.2):
    """
    Split the imbalanced data into training and validation sets in a balanced way by splitting the minority and majority subsets separately.
    """
    # Minority data
    idx_min = np.where(regions == 0)[0]
    X_min, y_min = X[idx_min], y[idx_min]
    X_min_train, X_min_val, y_min_train, y_min_val = train_test_split(
        X_min, y_min, test_size=test_size, random_state=42)
    
    # Majority data
    idx_maj = np.where(regions == 1)[0]
    X_maj, y_maj = X[idx_maj], y[idx_maj]
    X_maj_train, X_maj_val, y_maj_train, y_maj_val = train_test_split(
        X_maj, y_maj, test_size=test_size*idx_min.shape[0]/idx_maj.shape[0], random_state=42)
    
    X_train = np.vstack((X_min_train, X_maj_train))
    y_train = np.vstack((y_min_train, y_maj_train))
    X_val = np.vstack((X_min_val, X_maj_val))
    y_val = np.vstack((y_min_val, y_maj_val))
    regions_train = np.concatenate((np.ones(len(X_min_train))-1, np.ones(len(X_maj_train))))
    return X_train, y_train, X_val, y_val, regions_train

# -------------------------------
# Train and Evaluate Multi-output Regression Model
# -------------------------------
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train a multi-output regressor (RandomForest wrapped in MultiOutputRegressor) and compute MSE on validation and test sets.
    """
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1,random_state=42)
    model.fit(X_train, y_train)
    
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    mse_val = mean_squared_error(y_val, y_val_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    return model, mse_val, mse_test, y_val_pred, y_test_pred

# -------------------------------
# Plotting Helper Function
# -------------------------------
def plot_results(X, y, title=""):
    """
    Plot scatter of first feature vs. first response dimension.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], y[:, 0], alpha=0.5)
    plt.xlabel("Feature x1")
    plt.ylabel("Response y1")
    plt.title(title)
    plt.show()

def train_conditional_diffusion(model, X, regions, sampler, num_epochs=100, batch_size=256, seed=111, device='cuda'):
    set_seed(seed)
    """
    Train the conditional diffusion model on data (combined X and Y) with the condition as region.
    We simulate diffusion training by adding noise to the clean data and training the model to predict the noise.
    
    Args:
        model: an instance of ConditionalDiffusionModel.
        X: numpy array of shape (n, text_dim) with combined (X, Y) data.
        regions: numpy array of shape (n,) containing region labels.
        num_epochs: number of epochs.
        batch_size: batch size.
        device: 'cpu' or 'cuda'.
    """
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Create dataset and dataloader
    X_tensor = torch.tensor(X, dtype=torch.float32)
    regions_tensor = torch.tensor(regions, dtype=torch.float32).unsqueeze(-1)  # shape: [n,1]
    dataset = TensorDataset(regions_tensor,X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    n_samples = X.shape[0]
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in dataloader:
            image_embeddings, text_embeddings = data
            x = image_embeddings.to(device).float()
            y = text_embeddings.to(device).float()
            # image_embeddings are already precomputed and moved to device in __getitem__
            batch_size = text_embeddings.size(0)

            t = sampler.sample_timesteps(y.shape[0]).to(sampler.device)
            y_t, noise = sampler.noise_images(y,t.view(-1,1))
            predicted_noise = model(y_t, x, t)

            # Compute loss
            loss = criterion(predicted_noise, noise)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item() * batch_size
        
        epoch_loss = running_loss / n_samples
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    return model


class Autoencoder(nn.Module):
    def __init__(self, input_dim=5, latent_dim=3, hidden_dim=64):
        super(Autoencoder, self).__init__()
        # Encoder: from input_dim to latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Decoder: from latent_dim back to input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    
def train_autoencoder(autoencoder, XY, num_epochs=50, learning_rate=1e-3, device='cuda'):
    set_seed(111)
    autoencoder = autoencoder.to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    autoencoder.train()
    
    X_tensor = torch.tensor(XY, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)   
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            recon, _ = autoencoder(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    return autoencoder


def generate(model,decoder, m,alpha,sampler,seed=111):
    sythetic_Y=[0]*int(m*alpha)+ [1]*int(m*(1-alpha))
    regions = torch.tensor(sythetic_Y).float().view(-1,1)#F.one_hot(torch.tensor(sythetic_Y).type(torch.LongTensor),num_classes=2).float()
    X_syn_full, y_syn_full = generate_synthetic_data(regions, model, decoder, sampler, text_dim= model.out_fc.out_features ,num_synthetic=m,seed=seed)
    return X_syn_full, y_syn_full