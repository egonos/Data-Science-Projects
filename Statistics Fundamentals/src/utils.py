import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
import pandas as pd
import numpy as np

def compute_vif(df):
    vif_df = pd.DataFrame()
    vif_df["feature"] = df.columns
    vif_df["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_df

def likelihood_ratio_test(full_model, reduced_model):
    LL_full = full_model.llf
    LL_reduced = reduced_model.llf

    df_diff = full_model.df_model - reduced_model.df_model
    LR_stat = 2 * (LL_full - LL_reduced)
    p_value = stats.chi2.sf(LR_stat, df_diff)

    print(f"LR stat: {LR_stat:.3f}, df diff: {df_diff}, p-value: {p_value:.4f}")
    return LR_stat, p_value


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))


def check_linearity(x, y, df, q = 4):
    df_temp = df.copy()
    
    df_temp["x_bin"] = pd.qcut(df_temp[x], q=q, duplicates='drop')

    summary = df_temp.groupby("x_bin").agg(
        median_x = (x, "median"),
        p_hat = (y, "mean")
    ).reset_index()

    summary["p_hat"] = summary["p_hat"].clip(0.01, 0.99)
    summary["log_odds"] = np.log(summary["p_hat"] / (1 - summary["p_hat"]))

    plt.figure()
    plt.scatter(summary["median_x"], summary["log_odds"], color='steelblue')
    plt.plot(summary["median_x"], summary["log_odds"], linestyle='--', color='gray')
    plt.xlabel(f"Median of {x}")
    plt.ylabel("Logit(p)")
    plt.title(f"{x} vs Log odds of {y}")
    plt.grid(True)
    plt.show()

    return summary