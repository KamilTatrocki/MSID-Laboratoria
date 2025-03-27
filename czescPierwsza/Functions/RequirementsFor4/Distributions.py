import os

import seaborn as sns
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))

def plot1(data):
    sns.displot(data=data, x='overall', binwidth=1, kde=True)
    plt.title("Rozkład oceny ogólnej (overall)")
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'Plots', 'Distribution1.png'))
    plt.tight_layout()
    plt.savefig(file_path)

def plot2(data):
    plt.figure(figsize=(10, 12))
    sns.displot(data=data, x='work_rate', binwidth=1, )
    plt.title("Rozkład work rate ")
    plt.xticks(rotation=45)
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..','Plots','Distribution2.png'))
    plt.tight_layout()
    plt.savefig(file_path)

def plot3(data):
    sns.displot(data=data, x='weight_kg', binwidth=5, kde=True)
    plt.title("Rozkład wagi zawodników (weight_kg)")
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..','Plots','Distribution3.png'))
    plt.tight_layout()
    plt.savefig(file_path)

def plot4(data):
    sns.displot(data=data, x='height_cm', binwidth=4, kde=True)
    plt.title("Rozkład wzrostu zawodników (height_cm)")
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..','Plots','Distribution4.png'))
    plt.tight_layout()
    plt.savefig(file_path)







def SaveAllPlots(data1):
    plot1(data1)
    plot2(data1)
    plot3(data1)
    plot4(data1)

