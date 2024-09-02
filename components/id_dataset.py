import sys

from sklearn.preprocessing import (StandardScaler, LabelEncoder)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler



class IDDataset():  # Using a tf.Dataset object

    def __init__(self, label_name: str = None, path: str = None, df: pd.DataFrame = None,
                 features: list = None):

        self.df = {}

        if df is not None:
            self.df['original'] = df.copy(deep=True)

        # The path is linked to a dataset
        else:
            self.df['original'] = pd.read_csv(path, names=features, header=0, index_col=0)

        # Columns format
        self.df['original'].columns = self.df['original'].columns.astype(str)
        self.df['original'].columns = self.df['original'].columns.str.strip()

        self.df['original'] = self.df['original'].sample(frac=1)

        self.df['preprocessed'] = self.df['original'].copy(deep=True)

        self.label_name = label_name
        self.name = path.split("\\")[-1].replace('.csv','') if path else "IDDataset instance"

    @classmethod
    def get_test_instance(cls):
        """
        Return a basic instance for test or simple manipulation
        """
        df = pd.DataFrame({'class': ['A', 'A', 'B', 'C'],
                           'col2': ['X', 'X', 'Y', 'Y'],
                           'col3': [3.0, 4.0, None, 2.0],
                           'col4': [10.0, np.inf, 1.0, -np.inf]},
                          index=['row1', 'row2', 'row3', 'row4'])

        return cls('class', df=df)

    # === Core methods === #

    def preprocess(self) -> None:
        """
        Run every preprocessing methods.
        This method is used in the main workflow when dealing with IDFlows
        """

        # Drop irrelevant features for intrusion detection
        features_to_drop = ['application_is_guessed', 'application_confidence', 'requested_server_name',
                            'client_fingerprint', 'server_fingerprint', 'user_agent', 'content_type',
                            'src_mac', 'dst_mac', 'application_name']

        self.df['preprocessed'] = self.df['preprocessed'].drop(labels=features_to_drop, axis=1)

        # Preprocess methods
        self.encode_features(
            lab_enc=['src_ip', 'dst_ip', 'src_oui', 'dst_oui']
        )
        self.standardization_df()
        self.reduce_dimensions()

    def encode_features(self, one_hot_enc_: list = [], lab_enc: list = []) -> None:
        """
        Encode categorical data into integer
        """

        lab_enc = [self.label_name] + lab_enc if (self.label_name) else lab_enc

        one_hot_enc = [elt for elt in
                       self.df['preprocessed'].select_dtypes(include=["object"])
                       if elt not in lab_enc] + one_hot_enc_
        one_hot_enc = list(set(one_hot_enc))

        # Label encoder
        for col in lab_enc:
            lab_encoder = LabelEncoder()
            self.df['preprocessed'][col] = lab_encoder.fit_transform(self.df['preprocessed'][col])

        # One hot encoding
        self.df['preprocessed'] = pd.get_dummies(self.df['preprocessed'], columns=one_hot_enc, prefix="", prefix_sep="")

    def standardization_df(self):
        std_scaler = StandardScaler()  # z = (x - u) / s
        col = list( self.df['preprocessed'].select_dtypes(include='number').columns)

        if self.label_name in col:
            col.remove(self.label_name)

        for i in col:
            arr = self.df['preprocessed'][i]
            arr = np.array(arr)
            self.df['preprocessed'][i] = std_scaler.fit_transform(arr.reshape(len(arr), 1))

        return self.df['preprocessed']

    def min_max_normalization(self) -> pd.DataFrame:
        """
        Normalize the dataset using the min-max method

        Comments: Leads to very (very) small values due to the important variance -> rescale ?
        """
        df = self.df['preprocessed']

        for column in df.columns:
            column_min = float(df[column].min())
            column_max = float(df[column].max())

            if (column_min != column_max):
                df[column] = (df[column] - column_min) / (column_max - column_min)
            else:
                df[column] = 0

        return df

    def undersample(self, col_name: str, df_name: str = 'original', inplace: bool = True) -> pd.DataFrame:
        """
        Undersample a dataframe column

        ### Parameters
            - ``df_name (str)``: Name of the dataframe to undersample - default='original'
            - ``col_name (string)``: Column name to undersample
            - ``inplace (bool)``: If set to True, apply the changes to the class. Else, returns the modified dataframe.

        ### Returns
            - ``df_resampled (pd.DataFrame)``: Undersampled dataframe

        """

        rus = RandomUnderSampler()
        try:
            x, y = self.df[df_name].drop(col_name, axis=1), self.df[df_name][col_name]
        except KeyError as e:
            sys.exit(f"The dataframe '{e}' was not found.")

        print('x: ', x, 'y: ', y)
        x_resampled, y_resampled = rus.fit_resample(x, y)
        df_resampled = pd.DataFrame(x_resampled, columns=x_resampled.columns)
        df_resampled[col_name] = y_resampled

        if inplace:
            self.df[df_name] = df_resampled

        return df_resampled

    def drop_correlated_features(self) -> None:
        """
        Drop columns that are highly correlated (> 95%)
        """

        # Drop highly correlated features cf https://medium.com/@subrata.maji16/building-an-intrusion-detection-system-on-unsw-nb15-dataset-based-on-machine-learning-algorithm-16b1600996f5

        num_features = len(self.df["preprocessed"].columns)
        corr_matrix = self.df["preprocessed"].corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )  # Select upper triangle of correlation matrix

        features_to_drop = [
            column for column in upper.columns if any(upper[column] > 0.95)
        ]  # Find index of feature columns with correlation greater than 0.95

        self.df["preprocessed"] = self.df["preprocessed"].drop(features_to_drop, axis=1)
        print(
            f"...I kept {len(self.df['preprocessed'].columns)} non-collerated features over {num_features}"
        )

    def reduce_dimensions(self, nb_dim_: int = None, min_variance: float = 0.97,
                          inplace: bool = True) -> (pd.DataFrame, pd.DataFrame):
        """
        Reducing  dimension using PCA

        ### Parameters
          nb_dim_ (int): Number of dimensions to keep (None)
          min_variance (float): Minimum cumulated variance needed among the kept dimension(s) (0.97)
          inplace (bool): Whether to modify the DataFrame rather than returning a new one. (True)

        ## Returns
          reduced_df (DataFrame): Dataframe with reduced dimensions
          pca (PCA): PCA obj

        """

        #### remove label

        labels = self.df['preprocessed'][self.label_name].tolist()
        df_without_label = self.df['preprocessed'].drop(labels=[self.label_name], axis=1, inplace=False)
        pca = PCA()
        pca.fit(df_without_label)

        # Get the number of dimensions to keep
        if not nb_dim_:
            individual_variances = pca.explained_variance_  # Variance expliquée par chaque composante principale
            cumulative_variances = np.cumsum(individual_variances) / np.sum(individual_variances)  # Variance totale
            nb_dim = len(self.df['preprocessed'].columns)

            for i, elt in enumerate(cumulative_variances):
                if elt > min_variance:
                    nb_dim = i + 1
                    break

        else:
            nb_dim = nb_dim_

            # Reduce dimensions
        pca = PCA(n_components=nb_dim)

        reduced_df = pd.DataFrame(pca.fit_transform(df_without_label))
        print(f".. I kept {len(reduced_df.columns)} dimensions over {len(self.df['preprocessed'].columns)}")

        # Component explanation
        comp_expl = pd.DataFrame(pca.components_, columns=df_without_label.columns, index=[f'PC-{i + 1}'
                                                                                           for i in range(nb_dim)])
        comp_expl = comp_expl.abs().loc[:, (comp_expl != 0).any(axis=0)]
        comp_expl = comp_expl.reindex(comp_expl.mean().sort_values(ascending=False).index, axis=1)
        self.pca_indiv_variance = pca.explained_variance_ratio_

        if self.label_name:
            reduced_df[self.label_name] = labels

        if inplace:
            self.df['reduced'] = reduced_df

        return reduced_df, pca

    def remove_invalid_val(self, k: int = 10, inplace: bool = True) -> pd.DataFrame:
        """
        Handle the following invalid values:
            - Infinite values are replaced by k times the max/min value and Nan
            - NaN values are replaced by the mean of the column

        ### Parameters
            k (int): The inf values will be replaced by k * times the max / min value - default=10
            inplace (bool): If set to True, apply the changes to the class. Else, returns the modified dataframe.


        ### Return
            df (DataFrame): Dataframe without inf values
        """

        df = self.df['preprocessed']

        for col in df.columns:
            if df[col].dtype.kind in 'bifc':  # We go through column containing numeric values

                # Replace -inf with -k * abs max value
                max_val = df.loc[~df[col].isin( [np.inf, -np.inf]), col].abs().max()
                df.loc[df[col] == -np.inf, col] = -k * max_val

                # Replace inf with k * max value
                df.loc[df[col] == np.inf, col] = k * max_val

                # For numeric columns, Nan are replaced by the mean
                replaced_nan_val = df.loc[~df[col].isin([np.inf, -np.inf, np.nan]), col].mean()

            else:
                # For obj columns, Nan are replaced by the mode
                replaced_nan_val = df[col].mode()[0]

            df.loc[df[col].isna(), col] = replaced_nan_val

        if inplace:
            self.set_df(df)

        return df


    # ====== Utils ====== #

    def display_classes(self, chosen_df: str = 'orignal') -> None:
        """
        Display the repartition between class.
        """

        chosen_df = chosen_df if chosen_df in self.df.keys() else 'original'
        label_counts = self.df[chosen_df][self.label_name].value_counts()

        label_counts.plot(kind='bar')
        plt.xlabel('Flow type')
        plt.ylabel('Number of flows')
        plt.title('Flow types repartition')
        plt.show()

    def split(self, shuffle: bool = False) -> (np.array, np.array, np.array, np.array):
        """
        Split the dataset into X_train, X_test, Y_train, Y_test

        ### Parameters
          shuffle (Bool): Shuffle the dataset before splitting [False]
        """

        x, y = self.get_x_y(shuffle=shuffle)
        return train_test_split(x, y, test_size=0.20, random_state=42)

    def input_size(self) -> int:
        """
        Get the input dimension

        ### Return
          size (int): Dataset number of features
        """

        try:
            size = len(self.df['reduced'].columns)
        except KeyError:
            size = len(self.df['preprocessed'].columns)

        if self.label_name:
            size -= 1  # Label doesn't belong to the input

        return size

    def change_label(self, labels_mapping: list) -> None:
        """
        ### Parameters
          labels_mapping: List of tuples [ ([former_labels ...], new_label), (..,..), .. ]
        """
        for former_labels, new_label in labels_mapping:
            self.df['preprocessed'][self.label_name] = self.df['preprocessed'][self.label_name].replace(former_labels,
                                                                                                        new_label)

        self.df['preprocessed'][self.label_name] = self.df['preprocessed'][self.label_name].astype('string')

    def sort_labels(self, ascending: bool = False) -> None:
        """
        Sort the dataset accorind to labels

        ### Parameters
          ascending (bool): Sort ascending vs. descending. - default: False
        """

        try:
            self.df['reduced'] = self.df["reduced"].sort_values(by=[self.label_name], ascending=ascending)
        except KeyError:
            self.df['preprocessed'] = self.df["preprocessed"].sort_values(by=[self.label_name], ascending=ascending)

    # ====== Getters ====== ##

    def  get_classes(self, df_name: str = "original") -> list:
        """
        Get the labels of the different classes in the dataset

        ### Parameters
          df_name (str): From which version of the dataset do we want to get the classes: original (default), reduced, preprocessed.

        ### Return
          (array of string) Labels of the classes
        """

        return self.df[df_name][self.label_name].unique().tolist()

    def get_data(self, original: bool = False) -> pd.DataFrame:
        """
        Get one of the dataframe among the original one, the preprocessed one and the reduced one.

        ### Return
          Dataframe
        """

        if original:
            return self.df['original']
        try:
            return self.df['reduced']
        except KeyError:
            return self.df['preprocessed']

    def get_x_y(self, shuffle: bool = False) -> (np.array, np.array):
        """
        Split the dataset into input and output for training

        ### Parameters
          shuffle (Bool): Shuffle the dataset before splitting [False]

        ### Return
          input (NP array): List of inputs
          output (NP array): List of labels
        """

        df = self.df['reduced'] if 'reduced' in list(self.df.keys()) else self.df['preprocessed']

        if shuffle:
            df = df.sample(frac=1)

        output = df[self.label_name].tolist()
        input_ = df.drop(labels=[self.label_name], axis=1)

        return np.asarray(input_).astype('float32'), np.asarray(output).astype('float32')
        # return input_, output

    def get_flows_2d(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Get flows with 2d only for visualization.
        Use PCA to keep the 2dim with more information.
        :return flows_2d: [Dataframe] Flows dataset with 2 features.
        :return comp_expl: [Dataframe] Contribution of every feature in the final components
        """

        flows_2d, comp_expl = self.reduce_dimensions(nb_dim_=2, inplace=False)
        flows_2d['bidirectional_duration_ms'] = self.df['original']['bidirectional_duration_ms']

        return flows_2d, comp_expl

    # ====== Setters ====== #

    def set_df(self, new_df: pd.DataFrame, df_name: str = "preprocessed") -> None:
        """
        Update the preprocessed dataframe.

        ### Parameters
          - ``new_df`` (DataFrame): The updated dataframe.
          - ``df_name`` (str): Dataframe to upddate
        """

        self.df[df_name] = new_df
