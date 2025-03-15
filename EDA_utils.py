import polars as pl
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def pl_valueCounts_perc(
    col: str,
    df: pl.DataFrame,
    round_digit=2
) -> pl.DataFrame:
    """
    Returns a polar dataframe with count of each categories of the column "col" as well as their percentages
    """

    df_agg = (
        df.group_by(col)
        .agg(
            [
                pl.len().alias("count")
            ]
        )
        .with_columns(
            (pl.col("count") * 100 / pl.sum("count"))
            .round(round_digit)
            .alias("percentage %")
        )
        .sort(by = "percentage %", descending = True)
    )

    return df_agg

def pl_barplot(
    col: str,
    df: pl.DataFrame,
    order= None,
    x_rot= 90,
    annotate_rot= 'horizontal',
    annotate_fontsize: float = 10,
    yscale = True,
    figsize= (10,4)) -> None:
    """
    Plot sns.barplot() for a polar dataframe where barplots are annotated and sorted based on counts
    """

    df_agg = pl_valueCounts_perc(
        col,
        df,
    )

    # plot the barplot
    plt.figure(figsize= figsize)
    s = sns.barplot(
        data= df_agg.to_pandas(),
        x= col,
        y= 'count'
    )

    # annotate the bar plots
    for i, p in enumerate(s.patches):

        # extract percentage values for annotation
        percentage = df_agg["percentage %"][i]

        s.annotate(f'{percentage} %',
            (p.get_x()+p.get_width()/2., p.get_height()),
            ha = 'center', va = 'center',
            xytext = (0,9),
            textcoords = 'offset points',
            fontsize = annotate_fontsize,
            rotation = annotate_rot
        )

    if yscale:
        plt.yscale('log')

    plt.xticks(rotation = x_rot)
    plt.show()


def return_VIF(df:pl.DataFrame, target_col: str)-> pd.DataFrame:
    # Define the predictor variables
    X = df.drop(target_col)

    # Add an intercept
    X = X.with_columns(intercept= pl.lit(1))

    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.to_numpy(), i) for i in range(X.shape[1])]
    return vif_data.sort_values("VIF", ascending=False)



def handle_outliers(df: pl.DataFrame, factor: float = 1.5, remove_outliers: bool = False) -> pl.DataFrame:
    """
    Flag potential outliers in a Polars DataFrame using the IQR method and optionally remove them.

    Parameters:
    - df (pl.DataFrame): The input DataFrame.
    - factor (float): The multiplier for the IQR to define the outlier thresholds.
    - remove_outliers (bool): If True, returns the DataFrame with rows containing outliers removed.
                              If False, returns the DataFrame with an added 'is_outlier' column.

    Returns:
    - pl.DataFrame: The processed DataFrame either with an 'is_outlier' flag column or with outlier rows removed.
    """
    # Identify numeric columns (float and int types)
    numeric_cols = [col for col, dtype in df.schema.items() if dtype in {pl.Float64, pl.Float32, pl.Int64, pl.Int32}]

    # Start with a boolean mask that assumes no outliers
    mask = pl.Series("is_outlier", [False] * df.height)

    # Iterate through each numeric column to update the mask for potential outliers
    for col in numeric_cols:
        q1 = df.select(pl.col(col).quantile(0.25)).item()
        q3 = df.select(pl.col(col).quantile(0.75)).item()
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        # Generate a boolean mask for the current column's outliers
        current_mask = df.select((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)).to_series()

        # Combine with the existing mask (row is flagged if any column is an outlier)
        mask = mask | current_mask

        if remove_outliers:
            # Return DataFrame with outlier rows removed
            df_clean = df.filter(~mask)
            return df_clean
        else:
            # Return DataFrame with an added 'is_outlier' column for flagging
            df_flagged = df.with_column(mask.alias("is_outlier"))
            return df_flagged


def remove_outliers_iqr(df: pl.DataFrame, factor: float = 1.5) -> pl.DataFrame:
    """
    Remove extreme outliers from numerical columns in a Polars DataFrame using the IQR method.

    Parameters:
    - df (pl.DataFrame): The input Polars DataFrame.
    - factor (float): The multiplier for the IQR to define the outlier thresholds.

    Returns:
    - pl.DataFrame: A DataFrame with extreme outliers removed.
    """
    # Identify numeric columns based on schema (adjust types as needed)
    numeric_cols = [col for col, dtype in df.schema.items() if dtype in {pl.Float64, pl.Float32, pl.Int64, pl.Int32}]

    df_clean = df.clone()
    for col in numeric_cols:
        # Calculate Q1 and Q3 for each column
        q1 = df_clean.select(pl.col(col).quantile(0.25)).item()
        q3 = df_clean.select(pl.col(col).quantile(0.75)).item()
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        # Filter rows that fall within the bounds for this column
        df_clean = df_clean.filter((pl.col(col) >= lower_bound) & (pl.col(col) <= upper_bound))

    return df_clean




