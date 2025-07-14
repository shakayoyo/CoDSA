import os
import re
import math
import pickle
import random
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from utils import set_seed, data_split


    # 🔹 **Modified Dataset Class**
class UTKFaceDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=1):
        super(SimpleCNN, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Max Pooling Layer (2x2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        # For input 64x64, after 4 poolings: 64 -> 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 256)

        # Dropout layer
        self.dropout = nn.Dropout(0.4)

        # Output Layer (Regression)
        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        # Convolutional Blocks + Max Pooling
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 32, 32] for 64x64 input
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 16, 16]
        x = self.pool(F.relu(self.conv3(x)))  # [batch, 128, 8, 8]
        x = self.pool(F.relu(self.conv4(x)))  # [batch, 256, 4, 4]

        # Flatten the tensor
        x = x.view(x.size(0), -1)  # [batch, 256*4*4 = 4096]

        # Fully Connected Layers
        x = F.relu(self.fc1(x))     # [batch, 256]
        x = self.dropout(x)
        x = F.relu(self.fc2(x))     # [batch, 256]

        # Output Layer for regression
        x = self.output(x)          # [batch, num_classes]
        return x

def dir2df(dir_path):
        # Pattern to capture "imgg{i}.png", e.g. "imgg1234.png" => i=1234
    pattern = re.compile(r"^imgg(\d+)\.png$")
    metadata = []
    for filename in os.listdir(dir_path):
        # Each filename should look like: {age}_{gender}_{ethnicity}_imgg{i}.png
        # Example: 25_0_2_imgg37.png
        try:
            # Split at underscores: [age, gender, ethnicity, remainder]
            parts = filename.split("_")
            age_str, gender_str, ethnicity_str = parts[:3]
            remainder = parts[3]  # e.g., "imgg37.png"

            age = int(age_str)
            gender = int(gender_str)
            ethnicity = int(ethnicity_str)

            # Parse out the index i from remainder (which should be "imgg{i}.png").
            # e.g. "imgg37.png" => i=37
            match = pattern.match(remainder)
            if not match:
                continue
            index_str = match.group(1)
            index_i = int(index_str)

            # Store metadata in a dictionary
            metadata.append({
                "image_path": filename,
                "age": age,
                "gender": gender,
                "ethnicity": ethnicity,
                "index": index_i
            })
        except ValueError:
            # If something doesn't parse correctly, skip the file
            pass

    # Convert to DataFrame
    df = pd.DataFrame(metadata)

    # Optional: create ethnicity label
    ethnicity_labels = ["White", "Black", "Asian", "Indian", "Other"]
    df["ethnicity_label"] = df["ethnicity"].map(lambda x: ethnicity_labels[x] if x < len(ethnicity_labels) else "Unknown")

    # Optional: create gender label
    df["gender_label"] = df["gender"].map(lambda x: "Male" if x == 0 else "Female")
    return df




def predict(
    train_dataloader,
    val_dataloader,
    test_dataloader,
    df_test,
    ethnicity_labels,
    patience: int,
    device: torch.device,
    model_save_path: str = None
    seed: int =666
):
    """
    Train a SimpleCNN with early stopping, save the best model, and evaluate RMSE per ethnicity.

    Args:
      train_dataloader, val_dataloader, test_dataloader: DataLoader objects
      df_test: pandas DataFrame with columns 'ethnicity_label' and 'age'
      ethnicity_labels: list of unique ethnicity labels to evaluate
      sample_size: used to generate a default model filename
      patience: epochs without improvement before early stopping
      device: torch.device('cuda' or 'cpu')
      model_save_path: optional path to save/load the best model

    Returns:
      y_true_np: np.ndarray of true ages
      y_pred_np: np.ndarray of predicted ages
      mse_list: list of RMSE per ethnicity
    """
    # Default model path
    set_seed(666)

    # Initialize model, optimizer, loss
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Training loop with early stopping
    for epoch in range(1, 101):
        model.train()
        train_loss = 0.0
        for images, ages in train_dataloader:
            images, ages = images.to(device), ages.to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_dataloader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, ages in val_dataloader:
                images, ages = images.to(device), ages.to(device)
                preds = model(images).squeeze(1)
                val_loss += criterion(preds, ages).item() * images.size(0)
        val_loss /= len(val_dataloader.dataset)

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print("✅ Model improved; saved new best model.")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print(f"⏹ Early stopping triggered at epoch {epoch}.")
                break

    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # Test set predictions
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, ages in test_dataloader:
            images, ages = images.to(device), ages.to(device)
            outputs = model(images).squeeze(1)
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(ages.cpu().numpy())

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    ethnicity_np = df_test['ethnicity_label'].to_numpy()

    # Compute RMSE per ethnicity
    mse_list = []
    for label in ethnicity_labels:
        mask = (ethnicity_np == label)
        if mask.any():
            rmse = math.sqrt(mean_squared_error(y_true_np[mask], y_pred_np[mask]))
            mse_list.append(rmse)
        else:
            mse_list.append(None)

    print(f"RMSE per ethnicity: {mse_list}")
    return mse_list
    
    
    
train_transform = transforms.Compose([
    transforms.Resize((128,128)),  # typical for ResNet
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])

    # 🔹 **Load all images into RAM first**
def load_images_to_memory(root_dir, image_paths, transform):
    images = []
    labels = []

    for img_name in image_paths:
        img_path = os.path.join(root_dir, img_name)

        # Extract labels
        age, gender, ethnicity, _ = img_name.split("_")
        age = int(age)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        image = transform(image)

        images.append(image)
        labels.append(torch.tensor(age, dtype=torch.float32))

    # Stack all images into a single tensor array
    images = torch.stack(images)
    labels = torch.stack(labels)

    return images, labels



parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, help="Random seed to use for splits & generation")

args = parser.parse_args()  # This parses the command-line arguments
seed = args.seed 

device="cuda"

set_seed(666)

#Base model    
results =[]
train_dir = f"{seed}/utkface_train"
val_dir = f"{seed}/utkface_train"
test_dir = f"{seed}/utkface_test"

# Load metadata
df_train = pd.read_csv(f"{seed}/utkface_train_data.csv")
df_test = pd.read_csv(f"{seed}/utkface_test_data.csv")

df_val = df_train.groupby("ethnicity").apply(lambda x: x.iloc[:500]).reset_index(drop=True)
df_train = df_train.drop(df_val.index).reset_index(drop=True)
# Define image transforms

transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()  
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor()  
])



# ✅ **Preload training and test images into memory**
train_images, train_labels = load_images_to_memory(train_dir, df_train['image_path'].tolist(), transform)
test_images, test_labels = load_images_to_memory(test_dir, df_test['image_path'].tolist(), test_transform)
val_images, val_labels = load_images_to_memory(val_dir, df_val['image_path'].tolist(), test_transform)


# 🔹 **Use Preloaded Data in Dataset & Dataloader**
train_dataset = UTKFaceDataset(train_images, train_labels)
test_dataset = UTKFaceDataset(test_images, test_labels)
val_dataset = UTKFaceDataset(val_images, val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

mse = predict(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    df_test=df_test,
    ethnicity_labels=ethnicity_labels,
    patience=30,          # number of epochs to wait before early stopping
    device=device,
    model_save_path= f"seed/non_utkface_model.pth"  
)


print(f"RMSE per ethnicity: {mse}")
results.append((df_train.shape[0],mse))




    

with open(f"{seed}/result.pkl", "wb") as f:
    pickle.dump(results, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed for splits and training")
    args = parser.parse_args()
    seed = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(train_dir):
        data_split(seed=seed)
    
    #Base
    # Paths & metadata
    train_dir = f"{seed}/utkface_train"
    val_dir   = f"{seed}/utkface_train"
    test_dir  = f"{seed}/utkface_test"
    df_train  = pd.read_csv(f"{seed}/utkface_train_data.csv")
    df_test   = pd.read_csv(f"{seed}/utkface_test_data.csv")

    # split off validation
    df_val = df_train.groupby("ethnicity").head(500).reset_index(drop=True)
    df_train = df_train.drop(df_val.index).reset_index(drop=True)

    # transforms
    train_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # preload
    train_imgs, train_lbls = load_images_to_memory(train_dir,
                                                   df_train["image_path"].tolist(),
                                                   train_tf)
    val_imgs, val_lbls     = load_images_to_memory(val_dir,
                                                   df_val["image_path"].tolist(),
                                                   test_tf)
    test_imgs, test_lbls   = load_images_to_memory(test_dir,
                                                   df_test["image_path"].tolist(),
                                                   test_tf)

    # dataloaders
    train_loader = DataLoader(UTKFaceDataset(train_imgs, train_lbls),
                              batch_size=64, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(UTKFaceDataset(val_imgs,   val_lbls),
                              batch_size=64, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(UTKFaceDataset(test_imgs,  test_lbls),
                              batch_size=64, shuffle=False,
                              num_workers=4, pin_memory=True)

    ethnicity_labels = ["White", "Black", "Asian", "Indian", "Other"]
    results = []

    # initial run
    mse_list = predict(
        train_loader, val_loader, test_loader,
        df_test, ethnicity_labels,
        patience=30,
        device=device,
        model_save_path=f"{seed}/utkface_model_initial.pth",
        seed=seed
    )
    results.append(("initial", mse_list))
    
    #CoDSA

    train_dir = f"fake"
    test_dir = f"{seed}/utkface_test"
    val_dir = f"{seed}/utkface_train"



    # Load metadata
    df = dir2df(train_dir) 
    df_test = pd.read_csv(f"{seed}/utkface_test_data.csv")


    df_temp = pd.read_csv(f"{seed}/utkface_train_data.csv").iloc[10000:,].reset_index(drop=True)
    df_val = df_temp.groupby("ethnicity").apply(lambda x: x.iloc[:500]).reset_index(drop=True)
    df_temp = df_temp.drop(df_val.index).reset_index(drop=True)

    # ✅ **Preload training and test images into memory**
    train_images, train_labels = load_images_to_memory(train_dir, df['image_path'].tolist(), transform)
    test_images, test_labels = load_images_to_memory(test_dir, df_test['image_path'].tolist(), test_transform)
    val_images, val_labels = load_images_to_memory(val_dir, df_val['image_path'].tolist(), test_transform)
    temp_images, temp_labels = load_images_to_memory(val_dir, df_temp['image_path'].tolist(), test_transform)


    # 🔹 **Modified Dataset Class**


    train_images = torch.cat([train_images, temp_images], dim=0)
    train_labels = torch.cat([train_labels, temp_labels], dim=0)    

    # 🔹 **Use Preloaded Data in Dataset & Dataloader**
    train_dataset = UTKFaceDataset(train_images, train_labels)
    test_dataset = UTKFaceDataset(test_images, test_labels)
    val_dataset = UTKFaceDataset(val_images, val_labels)


    #train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    print("✅ Training Data Loaded into RAM Successfully!")

    err= np.array(mse)
    q_k = (err)/sum(err)

    pk=np.array([sum(df_temp["ethnicity_label"] == ethnicity_labels[x])/df_temp.shape[0] for x in range(5)])


    for sample_size in [10000, 15000, 20000,25000,30000]:
        print(f"\nTraining with {sample_size} samples per ethnicity...\n")
        ratio = 7705*(q_k-pk)/(sample_size)+ q_k

        ix=pd.concat([
            df[df["ethnicity_label"] == ethnicity_labels[x]].sample(n=int(sample_size*ratio[x]), random_state=42, replace=False)
            for x in range(5)
        ]).index.tolist() + list(range(df.shape[0],train_images.shape[0]))

        train_dataset = UTKFaceDataset(train_images[ix,], train_labels[ix])


        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

        mse = predict(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            df_test=df_test,
            ethnicity_labels=ethnicity_labels,
            patience=30,          # number of epochs to wait before early stopping
            device=device,
            model_save_path= f"seed/utkface_model_{sample_size}.pth"  
        )

        results.append((sample_size,mse))

    # save results
    
    with open(f"{seed}/results.pkl", "wb") as f:
        pickle.dump(results, f)