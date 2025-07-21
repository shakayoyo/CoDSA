import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

import argparse
import json
import typing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy 
import math
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, cohen_kappa_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


def evaluate(predicted_ratings,true_ratings):
    assert len(predicted_ratings) == len(true_ratings), "Mismatch in number of predictions and true labels."

    # Convert to numpy arrays for sklearn
    y_true = np.array(true_ratings)
    y_pred = np.array(predicted_ratings)

    # 4. Cohen’s Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa: {kappa:.4f}")
    return kappa


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def run(seed,ratio,transfer):
    df_review=pd.read_csv('data/short_reviews.csv')
    set_seed(seed)

    target_review=df_review[df_review['useful']!=0]
    source_review=df_review[df_review['useful']==0]
    test_data = pd.concat([
        target_review[target_review['stars'] == star].sample(n=1000, random_state=seed)
        for star in target_review['stars'].unique()
    ])

    # Get the remaining data for training
    train_data = target_review.drop(test_data.index)
    train_X = train_data["text"].tolist()
    test_X = test_data["text"].tolist()
    train_Y = torch.tensor(train_data["stars"].tolist(), dtype=torch.long)-1.0
    test_Y = torch.tensor(test_data["stars"].tolist(), dtype=torch.long)-1.0
    if ratio<1:
        remain_X, generation_X, remain_Y, generation_Y = train_test_split(
            train_X, train_Y, test_size=ratio, stratify=train_Y, random_state=seed
        )
    else:
        remain_X=[]
        remain_Y=[]

    if ratio ==1 and transfer ==1:
        df_generated = pd.read_csv(f"fake/{seed}/generated_text_data.csv")
    if ratio <1 and transfer ==1:
        df_generated = pd.read_csv(f"fake/{seed}/generated_text_data_point5.csv")
    if ratio ==1 and transfer ==0:
        df_generated = pd.read_csv(f"fake/{seed}/nontransfer_generated_text_data.csv")
    if ratio <1 and transfer ==0:
        df_generated = pd.read_csv(f"fake/{seed}/nontransfer_generated_text_data_point5.csv")

    df_generated = df_generated.sort_values(by='label').reset_index(drop=True)
    if ratio<1:
        X_train = df_generated["generated_text"].values.astype('U').tolist()+remain_X
        X_test  = test_X
        Y_train = np.array(df_generated["label"].tolist()+remain_Y.tolist())
        Y_test  = test_Y.tolist()

    else:
        X_train = df_generated["generated_text"].values.astype('U').tolist()
        X_test  = test_X
        Y_train = np.array(df_generated["label"].tolist())
        Y_test  = test_Y.tolist()

    sample_sizes = list(range(50000,len(Y_train)+1,50000))
    results = []


    qk = 0.2
    pk = np.array([sum(train_Y == x).item() for x in range(5)])
    pk = pk / sum(pk)


    # Train and evaluate the model on different sample sizes
    for size in sample_sizes:
        # Ensure size does not exceed dataset length
        actual_size = min(size, len(X_train))
        al= qk+len(remain_X)/size*(qk-pk)

        num_perclass = np.floor(actual_size*al)

    # Generate indices per class
        ix = []
        for i in range(5):  # Assuming 6 classes (0 to 5)
            start_idx = i * 200000
            end_idx = start_idx + min(int(num_perclass[i]),200000)
            ix.extend(range(start_idx, end_idx))  # Efficiently append indices
        ix.extend(range(1000000,len(X_train)))
        # Select a subset of data
        X_train_sample = [X_train[x] for x in ix]
        Y_train_sample = Y_train[ix]n
        vectorizer = TfidfVectorizer(analyzer='word', 
                                   stop_words='english',
                                   ngram_range=(1, 2),
                                   lowercase=True,
                                   min_df=5,
                                   binary=False)

        clf_lr = LogisticRegression(penalty='l2',
                                tol=1e-4,
                                C=5.0,
                                fit_intercept=True,
                                class_weight='balanced',
                                random_state=0,
                                solver='lbfgs',
                                max_iter=100,
                                multi_class='auto',
                                verbose=1,
                                n_jobs=-1)
        X_train_sample = vectorizer.fit_transform(X_train_sample)
        # Fit the model
        clf_lr.fit(X_train_sample, Y_train_sample)

        # Predict on the full test set
        #Y_pred = clf_lr.predict(X_test)
        X_test = vectorizer.transform(test_X)

        # Calculate accuracy
        kappa1 = evaluate(clf_lr.predict(X_test),Y_test)

        num_perclass = np.floor(actual_size*pk)

        # Generate indices per class
        ix = []
        for i in range(5):  # Assuming 6 classes (0 to 5)
            start_idx = i * 200000
            end_idx = start_idx + min(int(num_perclass[i]),200000)
            ix.extend(range(start_idx, end_idx))  # Efficiently append indices
        ix.extend(range(1000000, len(X_train)))
        # Select a subset of data

        X_train_sample = [X_train[x] for x in ix]
        Y_train_sample = Y_train[ix]
        vectorizer = TfidfVectorizer(analyzer='word', 
                                   stop_words='english',
                                   ngram_range=(1, 2),
                                   lowercase=True,
                                   min_df=5,
                                   binary=False)

        clf_lr = LogisticRegression(penalty='l2',
                                tol=1e-4,
                                C=5.0,
                                fit_intercept=True,
                                #class_weight='balanced',
                                random_state=0,
                                solver='lbfgs',
                                max_iter=50,
                                multi_class='auto',
                                verbose=1,
                                n_jobs=-1)
        X_train_sample = vectorizer.fit_transform(X_train_sample)
        # Fit the model
        clf_lr.fit(X_train_sample, Y_train_sample)
        X_test = vectorizer.transform(test_X)


        # Calculate accuracy
        kappa2 = evaluate(clf_lr.predict(X_test),Y_test)

        # Store the result
        results.append((actual_size, kappa1, kappa2))


    vectorizer = TfidfVectorizer(analyzer='word', 
                                   stop_words='english',
                                   ngram_range=(1, 2),
                                   lowercase=True,
                                   min_df=5,
                                   binary=False)
    oX_train = vectorizer.fit_transform(train_X)
    X_test  = vectorizer.transform(test_X)
    oY_train = train_Y.tolist()
    Y_test  = test_Y.tolist()


    clf_lr = LogisticRegression(penalty='l2',
                            tol=1e-4,
                            C=5.0,
                            fit_intercept=True,
                            class_weight='balanced',
                            random_state=0,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='auto',
                            verbose=1,
                            n_jobs=-1)
    clf_lr.fit(oX_train,oY_train)

    kappa_origin = evaluate(clf_lr.predict(X_test),Y_test)
    
    clf_lr = LogisticRegression(penalty='l2',
                            tol=1e-4,
                            C=5.0,
                            fit_intercept=True,
                            #class_weight='balanced',
                            random_state=0,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='auto',
                            verbose=1,
                            n_jobs=-1)
    clf_lr.fit(oX_train,oY_train)

    kappa_origin2 = evaluate(clf_lr.predict(X_test),Y_test)

    results.append((len(X_train), kappa_origin, kappa_origin2))
    return results

if __name__ == "__main__":
    all_result=[]
    for ratio in [0.5,1]:
        for transfer in [1,0]:
            res=[]
            for seed in tqdm([42,55,66,77,88]):
                res.append(run(seed,ratio,transfer))
            all_result.append(res)  

    with open(f"result/result.pkl", "wb") as f:
        pickle.dump(all_result, f)