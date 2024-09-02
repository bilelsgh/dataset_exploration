"""
Various functions used in the different classes and algorithms.
"""

import yaml
import shutil
import os
import sys
from typing import Union, List
from random import random

sys.path.append("../../dataset_overview")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
plt.switch_backend('TkAgg')

from components.id_dataset import IDDataset


def absolute_diff(liste: list) -> float:
    """
    Get the absolute difference between the first and last element of an array
    Mainly (only) used for neural network outputs.

    ### Parameter
        liste (List)

    ### Return
        diff (float): absolute difference

    Todo: what if we deal with multilabel classification?
    """
    diff = abs(liste[0] - liste[-1])

    return diff


def biased_wheel(probas: list) -> int:
    """
        Get a random element from a list biased by probabilities

        Parameters 
            probas (list): List of weight
        """

    # if sum( probas ) > 1:
    #     sys.exit(f"Sum of proba is different from 1: {sum(probas)}")

    total = sum(probas)
    v = random() * total
    tmp = 0
    i = -1

    while tmp <= v:
        i += 1
        tmp += probas[i]

    return i


def check_label_order(l: list) -> list:
    order = []

    for elt in l:
        if not order or order[-1] != elt:
            order.append(elt)

    return order

# ====== Viz ====== #
# Add to IDDataset ?

def classes_apparition(df: pd.DataFrame, col_name: str = "class", display: bool = True) -> None:
    """
    Display a heat map showing the arrival order of the different classes of a dataset

    ### Parameters
        df (DataFrame): Dataframe containing flows
        col_name (string): Name of the column containing the class label
        display (bool): Decide to display or not the visualization - default=True
    """

    onehot_encoder = OneHotEncoder(sparse=False)
    values = df[col_name].to_numpy()
    values = values.reshape(-1, 1)
    ohe_value = onehot_encoder.fit_transform(values)

    # get classes
    categories = onehot_encoder.categories_[0]

    presence = pd.DataFrame(ohe_value.T,
                            columns=list(range(len(df))),
                            index=categories)

    # viz
    fig, axes = plt.subplots(figsize=(20, 10))

    sns.heatmap(presence, annot=False, cmap="coolwarm", ax=axes)
    axes.set_title('Class apparition in time')
    axes.set_xlabel('Steps')
    axes.set_ylabel('Flow type')

    if display:
        plt.tight_layout()
        plt.show()

    return fig


def features_load(dataset: IDDataset, display: bool = True, title: str="",
                  min_load: float=None) -> List[str]:
    """
    Display the features importance for every Principal Component (PC)

    ### Parameters
        dataset (IDDataset): Dataset to preprocess
        display (bool): Decide to display or not the visualization - default=True
        title (str): Title to display on the viz
        min_load (float): If not None, we keep only the features that have a load > min_load for at least one PC

    """

    _, pca_normal = dataset.reduce_dimensions(inplace=True)  # perform the PCA
    non_important_features = []

    # Extraction des charges (loadings)
    idxs = dataset.df['preprocessed'].columns.tolist()
    idxs.remove( dataset.label_name )
    loadings1 = pd.DataFrame(pca_normal.components_.T,
                             columns=[f'PC{i + 1}' for i in range(dataset.get_data().shape[1])][:-1],
                             index=idxs)

    # Constraint on load
    if min_load:
        non_important_features = loadings1[( abs(loadings1) < min_load).all(axis=1)].index.tolist()
        loadings1 = loadings1[( abs(loadings1) > min_load).any(axis=1)] # we remove the features with less than min_load importance for every PC
        print(f"{title} (size: {len(dataset.get_data())}) => useless features: {non_important_features}")

    fig, axes = plt.subplots(figsize=(20, 10))

    # Heatmap pour le premier dataset
    sns.heatmap(loadings1, annot=False, cmap="coolwarm", ax=axes, cbar=True, xticklabels=True, yticklabels=True)
    axes.set_title(f'Feature Loadings - {title}', fontsize=16)
    axes.set_xlabel('Principal Components', fontsize=14)
    axes.set_ylabel('Features', fontsize=14)

    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)

    if display:
        plt.tight_layout()
        plt.show()

    return non_important_features

def features_load_per_class(dataset: IDDataset, display: bool = True) -> List[str]:
    """
    Display the features importance for each class

    ### Parameters
        dataset (IDDataset): Dataset to preprocess
        display (bool): Decide to display or not the visualization - default=True

    ### Return
        (list): Features not important for every class
    """

    df = dataset.get_data()
    classes = dataset.get_classes('preprocessed')
    non_import_features = []

    for class_ in classes:
        class_df = df[ df[dataset.label_name] == class_ ]
        dataset.set_df(class_df)

        non_import = features_load(dataset, display, f"{dataset.name} | {class_}", 0.1)
        non_import_features.append( set(non_import) )
        print(f"..Class *{class_}* done!")

    return list( set.intersection( *non_import_features ) )



