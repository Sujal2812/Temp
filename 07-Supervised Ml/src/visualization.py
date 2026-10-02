import plotly.express as px


def show_distribution(
    df,
    column
):

    fig = px.histogram(

        df,

        x=column,

        title=f"{column} Distribution"

    )

    return fig


def show_correlation(
    df
):

    corr = (

        df

        .select_dtypes(
            include="number"
        )

        .corr()

    )

    fig = px.imshow(

        corr,

        text_auto=True

    )

    return fig


def show_actual_vs_prediction(
    actual,
    prediction
):

    fig = px.scatter(

        x=actual,

        y=prediction,

        labels={

            "x": "Actual",

            "y": "Prediction"

        }

    )

    return fig