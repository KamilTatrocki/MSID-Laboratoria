import os

import seaborn as sns
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))

def plot1(data):
    sns.displot(data=data, x="overall", hue="body_type", multiple="stack", shrink=1, palette="RdYlGn", height=6,
                aspect=2)
    plt.title("Rozkład atrybutu Overall z podziałem na Body Type")
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'Plots', 'DistributionWithParameter1.png'))
    plt.tight_layout()
    plt.savefig(file_path)

def plot2(data):
    sns.displot(data=data, x="dribbling", hue="skill_moves", multiple="stack", shrink=1, palette="RdYlGn", height=6,
                aspect=2)
    plt.title("Rozkład dryblingu w zależności od Skill Moves")
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..','Plots','DistributionWithParameter2.png'))
    plt.tight_layout()
    plt.savefig(file_path)

def plot3(data):
    sns.displot(data=data, x="age", hue="preferred_foot", multiple="dodge", shrink=1, palette="RdYlGn", height=6,
                aspect=2)
    plt.title("Rozkład wieku ze względu na preferowaną nogę")
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..','Plots','DistributionWithParameter3.png'))
    plt.tight_layout()
    plt.savefig(file_path)
def plot4(data):
    filtered_data = data[data['pos_group'] != 'Inne']
    sns.displot(data=filtered_data, x="mentality_aggression", hue="pos_group", shrink=1, palette="Set1", height=6,
                aspect=1.5)
    plt.title("Rozkład atrybutu agresji w zależności od roli na boisku")
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..','Plots','DistributionWithParameter4.png'))
    plt.tight_layout()
    plt.savefig(file_path)






def SaveAllPlots(data1):
    plot1(data1)
    plot2(data1)
    plot3(data1)
    plot4(data1)

