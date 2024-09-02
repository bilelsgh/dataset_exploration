import sys
import time
sys.path.append('../dataset_overview')

import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

from components.id_dataset import IDDataset
from helpers.utils import classes_apparition, features_load

st.set_page_config(layout="wide")
st.title('Dataset preparation')

l0, r0 = st.columns(2)
label_name, dataset = None, None

# Data upload
with l0:
    st.markdown("")
    with st.expander('Init'):
        dataset_csv = st.file_uploader('Upload your dataset', type='csv')
        if dataset_csv:
            dataset_df = pd.read_csv(dataset_csv)
            label_name_for_df = st.text_input('Class label', placeholder='Name of the column containing the class label')
            label_name = label_name_for_df.strip()


# Overview metrics
delta_nb_lines, delta_nb_features, delta_nb_classes = 0, 0, 0
with r0:
    try:
        nb_lines, nb_features,nb_classes = len(dataset_df), len(dataset_df.columns), len(dataset_df[label_name_for_df].unique())
        col1, col2, col3 = st.columns(3)
        col1.metric('Number of line', nb_lines)
        col2.metric('Number of features', nb_features)
        col3.metric('Number of classes', nb_classes)
    except (KeyError, NameError) as e:
        pass

# IDDataset allows to perform preprocessing methods
if label_name:

    dataset = IDDataset(label_name=label_name, df=dataset_df)

    # Class distribution
    try:
        classes = dataset.get_data(True).groupby([label_name]).size().reset_index(name='count')
    except KeyError as e:
        st.warning(f'There is no column named "{e}" in the dataset. Please check the class label.')

    with st.expander('Overview'):
        st.dataframe(dataset.get_data(True))

    with st.expander('Statistics'):
        l1, r1 = st.columns(2)

        try:
            l1.markdown('##### Distribution of Classes')
            fig = px.pie(classes, values='count', names=label_name, title='')
            l1.plotly_chart(fig)
        except NameError as e:
            pass # We don't display the chart if 'classes' is not defined

        # Inf and normal values
        check_inf = dataset_df.isin( [np.inf, -np.inf] )
        inf_value = [ col for col in  check_inf.columns if check_inf[col].isin([True]).any() ]
        nan_value = [ col for col in  dataset_df.isna().columns if check_inf[col].isin([True]).any() ]

        inf_text = f"The following columns contain **infinite values**"
        nan_text = f"The following columns contain **NaN values**"
        r1.divider()
        r1.markdown( inf_text )
        r1.multiselect( "", inf_value, inf_value, disabled=True, key="inf_val" )
        r1.divider()
        r1.markdown( nan_text )
        r1.multiselect( "", nan_value, nan_value, disabled=True, key="nan_val" )
        st.divider()

        # Statistics
        st.markdown('##### Features statistics')
        st.dataframe(dataset_df.describe())

# Preprocessing
if dataset:

    st.markdown('---')
    st.subheader('Preprocessing')

    # Primary preprocessing functions
    bar = st.progress(0, 'Running preprocessing ')
    dataset.remove_invalid_val()
    bar.progress(25)
    dataset.standardization_df()
    bar.progress(50)
    dataset.encode_features()
    bar.progress(75)
    dataset.reduce_dimensions()
    bar.progress(100)
    time.sleep(1)
    bar.empty()


    # New datasets
    with st.expander("Preprocessed data"):
        preprocess_dataset_df = dataset.get_data()
        prep_nb_lines = len(preprocess_dataset_df)
        prep_nb_features = len(preprocess_dataset_df.columns)
        prep_nb_classes = len(preprocess_dataset_df[label_name].unique())

        # New metrics
        col1, col2, col3 = st.columns(3)
        col1.metric('Number of line', prep_nb_lines)
        col2.metric('Number of features', prep_nb_features)
        col3.metric('Number of classes', nb_classes)
        st.divider()

        # New datasets
        col4, col5 = st.columns(2)
        col4.markdown('##### Reduced data')
        col4.dataframe( preprocess_dataset_df )
        col5.markdown('##### Preprocessed data')
        col5.dataframe( dataset.df['preprocessed'] )

        # Viz
        col6, col7 = st.columns(2)
        with st.spinner("✨Beautiful visualizations will be displayed in a second✨"):
            col6.pyplot( classes_apparition(dataset.get_data(True), label_name, False) )
            col7.pyplot( features_load(dataset, False ))



