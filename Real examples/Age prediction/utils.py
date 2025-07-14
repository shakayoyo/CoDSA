import os
import pandas as pd
import random
from shutil import copy
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def data_download():
    # Authenticate and download dataset
    api = KaggleApi()
    api.authenticate()

    # Download and extract
    dataset_name = "jangedoo/utkface-new"
    output_dir = "utkface"
    api.dataset_download_files(dataset_name, path=output_dir, unzip=True)

def data_split(data_dir = "utkface/UTKFace", seed=111):
    train_dir = f"{seed}/utkface_train"
    test_dir = f"{seed}/utkface_test"
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Initialize storage for metadata
    metadata = []

    # Iterate through all image files and extract labels
    for filename in os.listdir(data_dir):
        try:
            # Extract labels from filename
            age, gender, ethnicity, _ = filename.split("_")
            age, gender, ethnicity = int(age), int(gender), int(ethnicity)

            # Store metadata
            metadata.append({"image_path": filename, "age": age, "gender": gender, "ethnicity": ethnicity})
        except:
            pass  # Skip files that do not match the expected format

    # Convert to DataFrame
    df = pd.DataFrame(metadata)

    # Define ethnicity labels
    ethnicity_labels = ["White", "Black", "Asian", "Indian", "Other"]
    df["ethnicity_label"] = df["ethnicity"].map(lambda x: ethnicity_labels[x])

    # Define gender labels
    df["gender_label"] = df["gender"].map(lambda x: "Male" if x == 0 else "Female")

    # Shuffle dataset
    df = pd.DataFrame(metadata)

    # Define ethnicity labels
    ethnicity_labels = ["White", "Black", "Asian", "Indian", "Other"]
    df["ethnicity_label"] = df["ethnicity"].map(lambda x: ethnicity_labels[x])

    # Define gender labels
    df["gender_label"] = df["gender"].map(lambda x: "Male" if x == 0 else "Female")

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Define the test dataset size per ethnicity
    test_size = 600

    # Storage for train and test datasets
    df_test = df.groupby("ethnicity").apply(lambda x: x.iloc[:test_size]).reset_index(drop=True)
    df_train = df[~df["image_path"].isin(df_test["image_path"])].reset_index(drop=True)

    # Step 1: Extract first 10,000 rows from original df_train
    df_train_head = df_train.iloc[:10000].reset_index(drop=True)

    # Step 2: Combine the rest of df_train with the existing df_test
    df_remaining = pd.concat([df_train.iloc[10000:], df_test], ignore_index=True)



    df_remaining = df_remaining.sample(frac=1, random_state=seed,replace=False).reset_index(drop=True)

    df_test_new = df_remaining.groupby("ethnicity").apply(lambda x: x.iloc[:test_size]).reset_index(drop=True)


    df_remaining_new_train = df_remaining[~df_remaining["image_path"].isin(df_test_new["image_path"])].reset_index(drop=True)
    df_train_new = pd.concat([df_train_head, df_remaining_new_train], ignore_index=True)

    duplicate_paths = df_train_new[df_train_new.duplicated("image_path", keep=False)]
    print(f"🔁 Number of duplicated image paths in df_train_new: {len(duplicate_paths)}")

    # Move images to respective folders
    for _, row in df_test_new.iterrows():
        copy(os.path.join(data_dir, row["image_path"]), os.path.join(test_dir, row["image_path"]))

    for _, row in df_train_new.iterrows():
        copy(os.path.join(data_dir, row["image_path"]), os.path.join(train_dir, row["image_path"]))

    # Save metadata
    df_train_new.to_csv(f"{seed}/utkface_train_data.csv", index=False)
    df_test_new.to_csv(f"{seed}/utkface_test_data.csv", index=False)

    print(f"✅ Train dataset: {len(df_train)} images")
    print(f"✅ Test dataset: {len(df_test)} images (600 per ethnicity)")

