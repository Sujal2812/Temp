import streamlit as st
import pandas as pd

from src.preprocess import load_data
from src.feature_engineering import create_features
from src.train import train

from src.visualization import (
    show_distribution,
    show_correlation,
    show_actual_vs_prediction
)


st.set_page_config(
    page_title="Pharma ML",
    layout="wide"
)

st.title(
    "💊 Drug Shelf-Life Prediction"
)


uploaded = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx"]
)

# IMPORTANT:
# df only exists AFTER upload

if uploaded is not None:

    df = load_data(
        uploaded
    )

    df = create_features(
        df
    )

    st.subheader(
        "Dataset"
    )

    st.dataframe(
        df.head()
    )

    num = (
        df
        .select_dtypes(
            include="number"
        )
        .columns
    )

    st.subheader(
        "Distribution"
    )

    if len(num):

        st.plotly_chart(

            show_distribution(
                df,
                num[0]
            )

        )

    st.subheader(
        "Correlation Matrix"
    )

    st.plotly_chart(

        show_correlation(
            df
        )

    )

    result, y, pred = train(
        df
    )

    report = pd.DataFrame(
        result,
        index=[
            "RMSE",
            "CV"
        ]
    )

    st.subheader(
        "Model Report"
    )

    st.dataframe(
        report
    )

    st.subheader(
        "Prediction Chart"
    )

    st.plotly_chart(

        show_actual_vs_prediction(
            y,
            pred
        )

    )

    st.success(
        "Completed"
    )