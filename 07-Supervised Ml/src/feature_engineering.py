import pandas as pd


def create_features(df):

    if (
        "Dosage" in df.columns
        and
        "Volume" in df.columns
    ):

        df[
            "dosage_ratio"
        ]=(
            df["Dosage"]
            /
            (
                df["Volume"]
                +1
            )
        )

    if "Temperature" in df.columns:

        df[
            "temp_band"
        ]=pd.cut(

            df["Temperature"],

            bins=[
                -100,
                10,
                25,
                50,
                100
            ],

            labels=[
                "Cold",
                "Normal",
                "Warm",
                "Hot"
            ]
        )

    return df