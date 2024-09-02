d   # What's in my data?

[Version: 1.0]
@bilel.saghrouchni

This application equips data scientists with essential tools for dataset exploration, cleaning, and preparation. It simplifies the extraction of insights and ensures data is ready for machine learning algorithms, streamlining the entire workflow and enhancing productivity.

## Project Structure

- `components/`: Contains Python classes used in this project.
      - `id_dataset.py`: Defines the IDDataset class for data preprocessing and manipulation.

- `dashboards/`: Simple and intuitive dashboards to visualize traffic data

- `helpers/`:
    - `data_preparation.py`: Preparation of every dataset used in the RL algorithm
    - `utils.py`: Some useful functions used in the project

  
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `requirements.txt`: Lists the project dependencies.
- `conf.yaml`: General parameters for the Q-Learning algorithm (dataset to use, number of episodes, learning rate etc.).
- `config.py`: Set env variables 



## Installation

1. Clone the repository:

```bash
  git clone https://github.com/X
  cd X
```
2. Install the libraries
```bash
pip install requirements.txt
```

## Run 

1. Set the parameters
```bash
conf.yaml
```

### Dashboard
2. Run:
```bash 
  streamlit run dashboards/data_prep.py
```


