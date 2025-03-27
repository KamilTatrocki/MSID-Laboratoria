import os

import seaborn as sns
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))

def plot1(data):
    filtered_data = data[data['body_type'] != 'Unique']
    plt.figure(figsize=(12, 6))

    sns.violinplot(x='body_type', y='weight_kg', data=filtered_data)
    plt.title("Waga zawodników (weight_kg) względem sylwetki")
    plt.xlabel("Typ sylwetki (body_type)")
    plt.ylabel("Waga (kg)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'Plots', 'ViolinPlot1.png'))
    plt.savefig(file_path)

def plot2(data):
    filtered_reputation_overall = data.dropna(subset=['international_reputation', 'overall'])
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='international_reputation', y='overall', data=filtered_reputation_overall)
    plt.title("Ocena ogólna (overall) względem Reputacji międzynarodowej (international_reputation)")
    plt.xlabel("Reputacja międzynarodowa (international_reputation)")
    plt.ylabel("Ocena ogólna (overall)")
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..','Plots','ViolinPlot2.png'))
    plt.tight_layout()
    plt.savefig(file_path)






def SaveAllPlots(data1):
    plot1(data1)
    plot2(data1)

