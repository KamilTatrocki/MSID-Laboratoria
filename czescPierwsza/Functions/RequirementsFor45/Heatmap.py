import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))

def plot1(data):
    cols = ["shooting", "attacking_finishing", "attacking_volleys", "attacking_heading_accuracy", "pace", "overall",
            "weak_foot", "skill_moves"]
    datafromcol = data[cols].apply(pd.to_numeric)
    corr_matrix = datafromcol.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmapa korelacji – Metryki napastnika")
    plt.tight_layout()
    plt.xticks(rotation=45)
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'Plots', 'HeatMap1.png'))
    plt.tight_layout()
    plt.savefig(file_path)

def plot2(data):
    cols = ["overall", "potential", "value_eur", "wage_eur"]
    datafromcol = data[cols].apply(pd.to_numeric)
    corr_matrix = datafromcol.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmapa korelacji – Metryki finansowe")
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..','Plots','HeatMap2.png'))
    plt.tight_layout()
    plt.savefig(file_path)

def plot3(data):
    cols = ["overall", "potential", "defending", "physic", "height_cm", "defending_marking_awareness",
            "defending_standing_tackle", "defending_sliding_tackle"]
    datafromcol = data[cols].apply(pd.to_numeric)
    corr_matrix = datafromcol.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmapa korelacji – Metryki obrońcy")
    plt.xticks(rotation=45)
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..','Plots','HeatMap3.png'))
    plt.tight_layout()
    plt.savefig(file_path)

def plot4(data):
    # %%
    cols = ["overall", "potential", "value_eur", "wage_eur"]
    datafromcol = data[cols].apply(pd.to_numeric)
    corr_matrix = datafromcol.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmapa korelacji – Metryki finansowe")
    plt.tight_layout()
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..','Plots','HeatMap4.png'))
    plt.tight_layout()
    plt.savefig(file_path)







def SaveAllPlots(data1):
    plot1(data1)
    plot2(data1)
    plot3(data1)
    plot4(data1)


