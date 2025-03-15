import numpy as np
import polars as pl
from scipy import stats
import polars.selectors as cs
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



def flag_outliers(
    df: pl.DataFrame,
    factor: float = 1.5,
    feature= None
    ) -> pl.DataFrame:
    """
    Flag potential outliers in a Polars DataFrame using the IQR method and z score.

    Parameters:
    - df (pl.DataFrame): The input DataFrame.
    - factor (float): The multiplier for the IQR to define the outlier thresholds.
    - feature: The desired numerical column

    Returns:
    - pl.DataFrame: The processed DataFrame either with 'Is_Zscore_Outlier' and 'Is_IQR_Outlier' flag columns.
    """

    df_flagged= df.clone()
    # -------------------------------
    # Outlier Detection using IQR
    # -------------------------------
    q1 = df_flagged.select(pl.col(feature).quantile(0.25)).item()
    q3 = df_flagged.select(pl.col(feature).quantile(0.75)).item()
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    df_flagged= df_flagged.with_columns(
        pl.when(
            (pl.col(feature)< lower_bound) | (pl.col(feature)> upper_bound)
        )
        .then(True)
        .otherwise(False)
        .alias("Is_IQR_Outlier")
    )
    # -------------------------------
    # Outlier Detection using Z-score
    # -------------------------------
    # Compute the z-scores for the chosen feature
    df_flagged= df_flagged.with_columns(
        z_score= np.abs(stats.zscore(df_flagged[feature]))
    )
    df_flagged= df_flagged.with_columns(
        pl.when(pl.col('z_score') >3)
        .then(True)
        .otherwise(False)
        .alias("Is_Zscore_Outlier")
    )

    # Scatter plot of z-scores to visualize anomaly detection
    plt.figure(figsize=(8, 6))
    # create index column
    df_flagged= df_flagged.with_row_index()
    plt.scatter(df_flagged["index"], df_flagged['z_score'], c=df_flagged['Is_Zscore_Outlier'], cmap="coolwarm", alpha=0.6)
    plt.axhline(3, color='red', linestyle='--', label='Z-score Threshold (3)')
    plt.title(f"Scatter Plot of Z-scores for {feature}")
    plt.xlabel("Index")
    plt.ylabel("Z-score")
    plt.legend()
    plt.show()

    # Summarize Outlier Counts
    iqr_outliers = df_flagged['Is_IQR_Outlier'].sum()
    zscore_outliers = df_flagged['Is_Zscore_Outlier'].sum()
    print(f"Number of IQR-based outliers with factor {factor} in '{feature}': {iqr_outliers}")
    print(f"Number of Z-score-based outliers in '{feature}': {zscore_outliers}")
    # filter data to only include potential outliers points
    df_flagged= df_flagged.filter(
        (pl.col("Is_IQR_Outlier")== True) | (pl.col("Is_Zscore_Outlier")== True)
    )[["index","Is_IQR_Outlier","Is_Zscore_Outlier"]]
    return df_flagged




