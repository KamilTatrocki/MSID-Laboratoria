import pandas as pd
def CalculateStatisticalFeature(data):
    numeric_stats = []
    categorical_stats = []

    for col in data.columns:
        missing_values = data[col].isna().sum()

        if pd.api.types.is_numeric_dtype(data[col]):
            col_mean = data[col].mean()
            col_median = data[col].median()
            col_min = data[col].min()
            col_max = data[col].max()
            col_std = data[col].std()
            col_5perc = data[col].quantile(0.05)
            col_95perc = data[col].quantile(0.95)

            numeric_stats.append({
                "column": col,
                "mean": col_mean,
                "median": col_median,
                "min": col_min,
                "max": col_max,
                "std": col_std,
                "5_percentile": col_5perc,
                "95_percentile": col_95perc,
                "missing_values": missing_values
            })

        else:
            unique_classes_count = data[col].nunique(dropna=True)
            value_counts = data[col].value_counts(dropna=True)

            total = value_counts.sum()
            class_proportions = (value_counts / total).to_dict() if total > 0 else {}

            categorical_stats.append({
                "column": col,
                "unique_classes_count": unique_classes_count,
                "missing_values": missing_values,
                "class_proportions": class_proportions
            })

    numeric_stats_df = pd.DataFrame(numeric_stats)
    categorical_stats_df = pd.DataFrame(categorical_stats)

    numeric_stats_df.to_csv("numeric_stats.csv", index=False, sep=";")
    categorical_stats_df.to_csv("categoricaal_stats.csv", index=False, sep=";")

