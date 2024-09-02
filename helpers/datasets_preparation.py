"""
Preparation of every datasets used in the
RL algorithm.
"""

import os
import sys

sys.path.append('../dataset_overview')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from enum import Enum

import numpy as np
import pandas as pd

from components.id_dataset import IDDataset


def select_dataset(name: str, path: str, **kwargs) -> IDDataset:
    """
    Run the function corresponding to the chosen dataset

    ### Parameters
        name (str): Dataset name
        path (str): Dataset path
        kwargs:
            - preprocess (bool): Set to True if the dataset need to be processed - default = True
            - undersampling (bool): Need to undersample the dataset (preprocess has to be set to True) - default=True
            - scenario (str)
    """
    datasets = {'cicids17': cicids,
                'nsl_kdd': nsl_kdd,
                "unsw_nb15": unsw_nb15}

    try:
        dataset_loader = datasets[name]
        dataset_args = {'path': path, **kwargs}

    except KeyError as e:
        sys.exit(f'(Error) No dataset found with the name "{e}". \
            Please choose one of the available dataset among {list(datasets.keys())}')

    return dataset_loader(**dataset_args)


def nsl_kdd(path, scenario: str = "EVERY_FLOW") -> IDDataset:
    """
    Preprocessing of the NSL-KDD dataset based on the IDDataset class' methods.

    ### Parameters
        path (string): Path of the dataset in CSV format
        preprocess (bool): Need to preprocess the dataset
        scenario (str): For which experimentation the dataset will be used
            - EVERY_FLOW - default: Keep every flow in the dataset
            - ATTACKS_ONLY: Keep attacks flows only
            - NORMAL_ONLY: Keep normal flows only
            - BIN_ATTACKS_APPARITION: Apparition of every attack after normal flows
            - MULTI_ATTACKS_APPARITION: Alternating normal flows and single attacks
        - undersampling (bool): Need to undersample the dataset (preprocess has to be set to True) - default=True

    ### Returns 
        - ```dataset (IDDataset)``
    """

    # scenario choice
    Scenario = Enum("Scenario",
                    ["EVERY_FLOW", "ATTACKS_ONLY", "NORMAL_ONLY", "BIN_ATTACKS_APPARITION", "MULTI_ATTACKS_APPARITION"])
    try:
        chosen_scenario = Scenario[scenario]
    except KeyError as e:
        sys.exit(f"{e} is not a valid scenario. Please chose one among: {' '.join(Scenario._member_names_)}")

        # labels processing
    feature = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
               "urgent", "hot",
               "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
               "num_file_creations", "num_shells",
               "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count",
               "serror_rate", "srv_serror_rate",
               "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
               "dst_host_count", "dst_host_srv_count",
               "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
               "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
               "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]

    binary_label = [(['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf',
                      'teardrop', 'udpstorm', 'worm', 'ftp_write', 'guess_passwd', 'httptunnel', 'imap',
                      'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'snmpguess', 'spy', 'warezclient',
                      'warezmaster', 'xlock', 'xsnoop', 'ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan',
                      'buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm'], 'attack')
                    # We're performing a binary classification
                    ]

    multi_label = [(['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop',
                     'udpstorm', 'worm'], 'Dos'),
                   (['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail',
                     'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop'], 'R2L'),
                   (['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'], 'Probe'),
                   (['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm'], 'U2R')
                   ]

    def preprocess_dataset(dataset_: IDDataset, label_mapping: list,
                           scenario_filter: int = None, sort_labels: bool = True) -> None:
        """
        Preprocess the Dataset according to different scenario

        ### Parameters
            dataset (IDDataset): Concerned dataset
            label_mapping (List[str]): Maps labels to binary or multi label
            scenario_filter (int): Parameter used to keep only one class in the dataset
            sort_labels (bool): If True (default) sorts the dataset  
        """

        dataset_.change_label(label_mapping)
        dataset_.encode_features(lab_enc=[dataset_.label_name], one_hot_enc_=['protocol_type', 'service', 'flag'])
        dataset_.standardization_df()

        if scenario_filter is not None:
            dataset_.set_df(dataset_.get_data()[dataset_.get_data()['label'] == scenario_filter])
        elif sort_labels:
            dataset_.sort_labels()

        # Retrieve data

    dataset = IDDataset(path=path, label_name="label", features=feature)

    print('\n..preprocessing the NSL KDD dataset')

    df = dataset.get_data()

    if chosen_scenario == Scenario.ATTACKS_ONLY:
        preprocess_dataset(dataset, binary_label, scenario_filter=0)

    elif chosen_scenario == Scenario.NORMAL_ONLY:
        preprocess_dataset(dataset, binary_label, scenario_filter=1)

    elif chosen_scenario == Scenario.BIN_ATTACKS_APPARITION:
        preprocess_dataset(dataset, binary_label)

    elif chosen_scenario == Scenario.MULTI_ATTACKS_APPARITION:
        preprocess_dataset(dataset, multi_label, sort_labels=False)

        flows = dataset.get_data()
        normal_flows = flows[flows['label'] == 4]
        splitted_normal_flows = np.array_split(normal_flows, 5)
        attack_labels = [0, 1, 2, 3]  # 0: ddos, 1: probe, 2: r2l, 3: u2r
        attacks_flows = [flows[flows['label'] == label] for label in attack_labels]

        formatted_dataset = pd.DataFrame()

        # alternate btw attacks and normal flows
        for i in range(5):
            temp = pd.concat([splitted_normal_flows[i], attacks_flows[i % 4]])
            temp = temp.sample(frac=1).reset_index(drop=True) # to get a more realistic behaviour
            formatted_dataset = pd.concat([temp, formatted_dataset])

        formatted_dataset.reset_index()

        dataset.set_df(formatted_dataset)

    elif chosen_scenario == Scenario.EVERY_FLOW:
        preprocess_dataset(dataset, binary_label, sort_labels=False)

        # Features selection
    dataset.reduce_dimensions(inplace=True) # PCA

    return dataset


def cicids(path, preprocess: bool = True, undersampling: bool = True):
    """
    Preprocessing of the CICIDS17 dataset based on the IDDataset class' methods.

    ### Parameters
        - path (string): Path of the dataset in CSV format
        - preprocess (bool): Need to preprocess the dataset - default=True
        - undersampling (bool): Need to undersample the dataset (preprocess has to be set to True) - default=True

    ### Returns
        - dataset (IDDataset): Preprocessed dataset
    """

    # Retrieve data
    dataset = IDDataset(path=path, label_name="Label")
    # print( f"Classes: {dataset.get_classes()}" )

    if preprocess:
        print('\n..preprocessing the CICIDS17 dataset')
        dataset.set_df(dataset.get_data().fillna(0))

        # Handle inf values
        print(dataset.get_data().columns.tolist())
        cols_containing_inf = ["Flow Packets/s", "Flow Bytes/s"]
        for col in cols_containing_inf:
            col_containing_inf = dataset.get_data()[col].unique().tolist()
            col_containing_inf.remove(np.inf)
            max_flow_bytes = max(col_containing_inf)
            dataset.get_data()[col].replace([np.inf],
                                            max_flow_bytes + 1,
                                            inplace=True)  # Flows that are equal to inf are replace by
            # the maximum value (without inf) + 1
        attacks_labels = dataset.get_data(original=True)[dataset.label_name].unique().tolist()
        try:
            attacks_labels.remove('BENIGN')
        except:
            pass
        dataset.change_label([(attacks_labels, 'ATTACK')])  # We're performing a binary classification

        dataset.standardization_df()
        dataset.encode_features(lab_enc=[dataset.label_name])
        # dataset.min_max_normalization()

        # Features selection
        dataset.reduce_dimensions()  # PCA: todo: add a fix number

        if undersampling:
            dataset.undersample(col_name=dataset.label_name, df_name="reduced")

    return dataset


def unsw_nb15(path):
    """
    Preprocessing of the UNSW-NB15 dataset based on the IDDataset class' methods.
    :param path: [string] Path of the dataset in CSV format
    :return x_train, x_test, y_train, y_test: [Dataframes or NP arrays] Training and testing inputs and labels.
    """
    # CHECK LABELS

    # Retrieve data
    dataset = IDDataset(path=path, label_name="label")
    print(f"\n..preprocessing the UNSW-NB15 dataset\n -> {len(dataset.get_data(original=True).columns)}\
          features\n -> {len(dataset.get_data(original=True))} records\n")
    # print(dataset.get_data(original=True).columns.tolist())
    # print(dataset.get_data(original=True).info())

    # Encode non numerical values
    dataset.standardization_df()
    dataset.encode_features(lab_enc=['attack_cat'], one_hot_enc_=['proto', 'service', 'state'])

    # Drop highly correlated features cf https://medium.com/@subrata.maji16/building-an-intrusion-detection-system-on-unsw-nb15-dataset-based-on-machine-learning-algorithm-16b1600996f5
    print('..features selection')
    num_features = len(dataset.get_data('preprocessed').columns)
    corr_matrix = dataset.get_data('preprocessed').corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))  # Select upper triangle of correlation matrix

    features_to_drop = [column for column in upper.columns if
                        any(upper[column] > 0.95)]  # Find index of feature columns with correlation greater than 0.9
    dataset.set_df(dataset.get_data('preprocessed').drop(features_to_drop, axis=1))
    print(f"...I kept {len(dataset.get_data('preprocessed').columns)} over {num_features}")

    # print(f"Went from {len( dataset.get_data(original=True).columns )} to {len( dataset.get_data('preprocessed').columns )}")

    # Features selection
    # dataset.reduce_dimensions() # PCA

    return dataset


if __name__ == "__main__":
    print("Main")
    data = select_dataset("nsl_kdd", "../datasets/NSL_KDD/KDDTrain+.csv")
    # data.get_data().to_csv("../datasets/CICIDS17/cicids17_preprocessed.csv")
    print("Done")
    print('End')
