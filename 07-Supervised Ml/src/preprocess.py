import pandas as pd


def load_data(file):

    file.seek(0)

    first_bytes = file.read(4)

    file.seek(0)

    try:

        # Excel files start with PK
        if first_bytes.startswith(b"PK"):

            df = pd.read_excel(
                file
            )

        else:

            try:

                df = pd.read_csv(
                    file
                )

            except:

                file.seek(0)

                df = pd.read_excel(
                    file
                )

    except Exception as e:

        raise ValueError(
            f"Cannot load dataset: {e}"
        )

    df.drop_duplicates(
        inplace=True
    )

    numeric = (

        df
        .select_dtypes(
            include="number"
        )
        .columns
    )

    categorical = (

        df
        .select_dtypes(
            exclude="number"
        )
        .columns
    )

    for c in numeric:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

        df[c] = df[c].fillna(
            df[c].median()
        )

    for c in categorical:

        if not df[c].mode().empty:

            df[c] = df[c].fillna(
                df[c].mode()[0]
            )

    df.reset_index(
        drop=True,
        inplace=True
    )

    return df