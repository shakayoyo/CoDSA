import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import random
import math
import copy


device ="cuda"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim, mlp_hidden_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = None
        if mlp_hidden_dim is not None:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, embed_dim)
            )
    
    def forward(self, t):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        if self.mlp:
            emb = self.mlp(emb)
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
    

def true_function(X):
    """
    Given X in R^3, returns y in R^2:
      y1 = sin(2*pi*x1) + x2,
      y2 = cos(2*pi*x3) + x1*x3.
    """
    #y1 = 2*np.sin(2 * np.pi * X[:, 0]) + np.exp(X[:, 1])
    #y2 = np.log(X[:, 0] * X[:, 2])*(X[:, 0]>0.5)-np.log(X[:, 0] * X[:, 2])*(1-(X[:, 0]>0.5))
    beta = np.array([[2.0], [-1.5], [3.0]])
    y_1 = X.dot(beta)+1 
    beta = np.array([[-2.0], [0.0], [2.0]])
    y_2 = X.dot(beta) 
    return np.column_stack([y_1,y_2])

def noise_function(X):
    """
    Computes a noise scale that depends on X.
    For example:
      scale = 0.2 + 0.5*sin(2*pi*x1) + 0.3*(x2)**2 + 0.1*x3.
    X is assumed to be of shape (n,3). Returns shape (n,1) to broadcast.
    """
    # Extract features: assume columns correspond to x1, x2, x3
    x1 = X[:, 0:1]
    x2 = X[:, 1:2]
    x3 = X[:, 2:3]
    #scale = (0.5 + 0.5 * (x2 ** 2) + 0.5 * x3)*(x1>=0.5)+(1 + 2 * (x2 ** 2) + 2 * x3)*(x1<0.5)
    scale = (0.5 )*(x1>=0)+(1)*(x1<0)
    #scale = (0.5 + 0.5 * (x2) + 0.5 * x3)*(x1>=0.5)+(0.5 + 2 * (x2 ** 2) + 2*(x3 ** 2))*(x1<0.5)
    return scale 

def combine_XY(X, y):
    """
    Concatenate X (R^3) and y (R^2) to form a 5-dimensional vector.
    """
    return np.hstack([X, y])

# -------------------------------
# Generate Imbalanced Data
# -------------------------------



def generate_imbalanced_data(n_minority, n_majority, seed=None):
    """
    Generate a high-dimensional, multi-modal classification dataset.
    
    Latent variables (u1, u2, u3) are drawn from mixture distributions:
      - u1: Mixture of two Gaussians (centered at -2 and 2)
      - u2: Mixture of two Uniform distributions (from [0,1] and [2,3])
      - u3: Shifted Exponential (exponential with scale=1, then shifted by -1)
      
    Twenty features are then constructed using various nonlinear transformations 
    that involve all three latent variables.
    
    The decision score is computed as:
    
         decision_score = sin(linear_comb) + 0.5*tanh(linear_comb) + 0.3*log(1+|linear_comb|)
    
    where linear_comb is a weighted sum of all 20 features. This ensures that the 
    decision function is related to all features.
    
    Finally, the minority class (labeled 0) is defined by selecting samples from both 
    the lowest and highest tails of the decision score distribution.
    
    Args:
      n_minority (int): Number of minority samples.
      n_majority (int): Number of majority samples.
      seed (int, optional): Random seed for reproducibility.
      
    Returns:
      X (np.array): Feature matrix of shape ((n_minority+n_majority), 20).
      Y (np.array): Binary labels (0 for minority, 1 for majority).
      regions (np.array): Region labels (identical to Y).
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_total = n_minority + n_majority
    
    # Generate latent variables with more complex distributions.
    # u1: Mixture of two Gaussians (50% chance each).
    mix_flag = np.random.rand(n_total) < 0.5
    u1 = np.where(mix_flag, np.random.randn(n_total) - 2, np.random.randn(n_total) + 2)
    
    # u2: Mixture of two Uniform distributions.
    mix_flag = np.random.rand(n_total) < 0.5
    u2 = np.where(mix_flag, np.random.uniform(0, 1, n_total), np.random.uniform(2, 3, n_total))
    
    # u3: Exponential distribution shifted to have both negative and positive values.
    u3 = np.random.exponential(scale=1.0, size=n_total) - 1.0

    # Construct 10 features using nonlinear transformations that involve all latent variables.

    f1  = u1 * u2
    f2  = u1 * u3
    f3  = u2 * u3
    f4  = u1**2
    f5  = u2**2
    f6  = u3**2
    f7 = u1*u2*u3
    f8 = u1**3
    f9 = u2**3
    f10 = u3**3

    X = np.column_stack([f1, f2, f3, f4, f5, f6, f7,
                         f8, f9,f10])
    
    # Compute a weighted linear combination of all features.
    # We assign weights that gradually increase from 0.5 to 1.5.
    weights = np.linspace(-1,1, 5)

    linear_comb1 = X[:,0:5].dot(weights)
    linear_comb2 = X[:,5:].dot(weights)
    
    
    #decision_score 
    tmp1= linear_comb1/(1+1*np.abs(linear_comb1))
    tmp2= linear_comb2/(1+1*np.abs(linear_comb2))
    
    decision_score = np.sin(2*np.pi*tmp1)**2- np.cos(3*np.pi*tmp2)**2
    # Ensure decision_score is a 1D array.
    decision_score = decision_score.flatten()
    
    # Multi-modal minority selection:
    # Select half of the minority samples from the lower tail and half from the upper tail.
    indices_sorted = np.argsort(decision_score)  # sorted in ascending order
    half_minority = n_minority // 2
    if n_minority % 2 == 0:
        minority_indices = np.concatenate([indices_sorted[:half_minority],
                                             indices_sorted[-half_minority:]])
    else:
        minority_indices = np.concatenate([indices_sorted[:half_minority+1],
                                             indices_sorted[-half_minority:]])
    
    # Create binary labels: 0 for minority, 1 for majority.
    Y = np.ones(n_total, dtype=int)
    Y[minority_indices] = 0
    
    # Region labels (for later use) are identical to Y.
    regions = Y.copy()
    
    return X, Y, regions


# -------------------------------
# Synthetic Data Generation for Minority Region
# -------------------------------
def generate_synthetic_data(regions, model, decoder, sampler, text_dim, num_synthetic=500, seed=111):
    set_seed(seed)
    """
    Generate synthetic data for the target (minority) region by oversampling with Gaussian perturbations.
    """
    test_dataloader = DataLoader(regions, batch_size=2048, shuffle=False)
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
    return generated_result,  regions

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



class BinaryClassificationNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=3):
        """
        A simple fully connected network for binary classification.
        Args:
            input_dim (int): Dimensionality of input.
            hidden_dim (int): Number of hidden units.
            output_dim (int): Dimensionality of output, typically 1 for binary classification.
            num_layers (int): Number of hidden layers.
        """
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # Return raw logits; BCEWithLogitsLoss applies the sigmoid internally.
        return self.net(x)

def binary_entropy(prob, eps=1e-12):
    """
    Computes binary entropy for a tensor of probabilities.
    H(p) = -p * log(p) - (1-p) * log(1-p)
    """
    prob = torch.clamp(prob, eps, 1 - eps)
    return -prob * torch.log(prob) - (1 - prob) * torch.log(1 - prob)

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, 
                       input_dim, hidden_dim=128, num_layers=3,
                       num_epochs=100, batch_size=32, learning_rate=1e-3, 
                       patience=10, device='cuda', seed = 42):
    """
    Train a binary classification neural network with early stopping based on validation loss.
    In addition to accuracy and predictive entropy, the model's BCE loss (BCEWithLogitsLoss) is computed 
    on the validation and test sets as a performance metric.
    
    Args:
        X_train, y_train: Training data (numpy arrays).
        X_val, y_val: Validation data.
        X_test, y_test: Test data.
        input_dim (int): Dimensionality of X.
        hidden_dim (int): Hidden layer size.
        num_layers (int): Number of hidden layers.
        num_epochs (int): Maximum number of epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        patience (int): Number of epochs with no improvement to wait before stopping.
        device (str): 'cpu' or 'cuda'.
    """
    device = torch.device(device)
    set_seed(42)
    # Convert numpy arrays to torch tensors; y_* are converted to float for BCEWithLogitsLoss.
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
    
    # Create training DataLoader.
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = BinaryClassificationNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = np.inf
    best_model_state = None
    epochs_no_improve = 0
    n_train = len(train_dataset)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        train_loss = running_loss / n_train
        
        # Evaluate on validation set.
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        #print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping based on validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model weights.
    model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        y_val_pred_logits = model(X_val_tensor)
        y_test_pred_logits = model(X_test_tensor)
    
    # Compute BCE metric on the raw logits.
    bce_val_metric = criterion(y_val_pred_logits, y_val_tensor).item()
    bce_test_metric = criterion(y_test_pred_logits, y_test_tensor).item()
    
    # Convert logits to probabilities using sigmoid.
    y_val_pred_prob = torch.sigmoid(y_val_pred_logits)
    y_test_pred_prob = torch.sigmoid(y_test_pred_logits)
    
    # Threshold probabilities at 0.5 to obtain binary predictions.
    y_val_pred = (y_val_pred_prob >= 0.5).float().cpu().numpy()
    y_test_pred = (y_test_pred_prob >= 0.5).float().cpu().numpy()
    

    print("Validation BCE Metric:", bce_val_metric)
    print("Test BCE Metric:", bce_test_metric)

    
    return model, bce_val_metric, bce_test_metric, y_val_pred_prob, y_test_pred_prob 


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
    def __init__(self, input_dim=5, latent_dim=3, hidden_dim=64, num_layer=3):
        """
        Parameters:
            input_dim (int): Dimensionality of the input data.
            latent_dim (int): Dimensionality of the latent representation.
            hidden_dim (int): Number of units in the hidden layers.
            num_layer (int): Total number of linear layers in the encoder/decoder.
                             Must be at least 2 (one hidden layer and one output layer).
        """
        super(Autoencoder, self).__init__()
        
        if num_layer < 2:
            raise ValueError("num_layer must be at least 2 to define the network layers.")

        # Build the encoder
        encoder_layers = []
        # First layer: input_dim -> hidden_dim
        encoder_layers.append(nn.Linear(input_dim, hidden_dim))
        encoder_layers.append(nn.ReLU())
        # Intermediate hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layer - 2):
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
        # Final layer: hidden_dim -> latent_dim
        encoder_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build the decoder
        decoder_layers = []
        # First layer: latent_dim -> hidden_dim
        decoder_layers.append(nn.Linear(latent_dim, hidden_dim))
        decoder_layers.append(nn.ReLU())
        # Intermediate hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layer - 2):
            decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
        # Final layer: hidden_dim -> input_dim
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
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
    regions = torch.tensor(sythetic_Y).float().view(-1,1)
    X_syn_full, y_syn_full = generate_synthetic_data(regions, model, decoder, sampler, text_dim= model.out_fc.out_features ,num_synthetic=m,seed=seed)
    return X_syn_full, y_syn_full