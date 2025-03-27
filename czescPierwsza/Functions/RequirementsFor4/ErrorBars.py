import os

import seaborn as sns
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))

def plot1(data):
    plt.figure(figsize=(14, 6))

    sns.pointplot(x='defending', y='mentality_aggression', data=data, errorbar=('ci', 95), err_kws={'color': 'red'})

    plt.xlabel("defending")
    plt.ylabel("mentality aggression")
    plt.title("Defending w zależności od mentality aggression z errorbarami")
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'Plots', 'ErrorBar1.png'))
    plt.tight_layout()
    plt.savefig(file_path)

def plot2(data):
    plt.figure(figsize=(10, 6))

    sns.pointplot(x='weight_kg', y='height_cm', data=data, errorbar=('ci', 95), err_kws={'color': 'red'})

    plt.xlabel("Waga (kg)")
    plt.ylabel("Wzrost (cm)")
    plt.title("Wzrost w zależności od wagi z errorbarami")
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..','Plots','ErrorBar2.png'))
    plt.tight_layout()
    plt.savefig(file_path)






def SaveAllPlots(data1):
    plot1(data1)
    plot2(data1)

