import os
from pathlib import Path
import sys

import yaml


HERE = Path(__file__).parent
CONF_PATH: Path = HERE / "conf.yaml"


if CONF_PATH.is_file():
    with open(CONF_PATH, 'r') as f:
        config = yaml.safe_load(f)

    try:
        # Global
        mlflow_uri = config['experimentations']['mlflow_uri']
        run_name= config['experimentations']['run_name']
        track= config['experimentations']['track']
        dataset = config['dataset']
        dataset_name = config['dataset']['name']
        dataset_path = str( HERE / config['dataset']['path'] )
        experiment_name = config['experimentations']['exp_name']
        preprocess_dataset = dataset['preprocess_dataset']
        undersampling = dataset['undersampling']
        nslkdd_scenario = dataset['scenario']

        # Clustering
        viz_path = HERE / Path(f"{config['clustering']['viz_path']}/{run_name}") \
            if config['clustering']['viz_path'] else None
        clustering_buffer_size = config['clustering']['buffer_size']
        k_ = config['clustering']['dynamic_number_clusters']
        percent_discard = config['clustering']['prop_discard']



        # Q-Learning
        state_size = config['qlearning']['state_size']
        num_iterations = config['qlearning']['num_iterations']
        num_episodes = config['qlearning']['num_episodes']
        copy_step = config['qlearning']['copy_step']
        train_step = config['qlearning']['train_step']
        replay_buffer_size = config['qlearning']['replay_buffer_max_size']
        replay = config['qlearning']['experience_replay']
        replay_buffer_type = config['qlearning']["replay_buffer_type"]
        log_path = config['experimentations']['dql_path']
        supervised_reward_function = config['qlearning']['supervised_reward_function']

        epsilon = config['qlearning']['epsilon']
        decoy_rate = config['qlearning']['decoy_rate']
        discount_factor = config['qlearning']['discount_factor']

        # Q-Network
        type_ = config['qnetwork']['type']
        units = config['qnetwork']['units']
        epochs = config['qnetwork']['epochs']
        batch_size_nn = config['qnetwork']['batch_size_nn']
        nb_hidden_layers = config['qnetwork']['nb_hidden_layers']
        models_path = config["experimentations"]["models_path"]
        loss_function = config['qnetwork']['loss']

        if viz_path and not viz_path.is_dir():
            os.mkdir( viz_path )


    except KeyError as e:
        sys.exit(f"The parameter '{e}' was not found. Please check {CONF_PATH}.")

