import os

import seaborn as sns
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))

def plot1(data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='club_position', y='height_cm', data=data)
    plt.title("Wzrost zawodników  względem pozycji w klubie")
    plt.xlabel("Pozycja w klubie")
    plt.ylabel("Wzrost (cm)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'Plots', 'BoxPlot1.png'))
    plt.savefig(file_path)

def plot2(data):
    filtered_data = data[data['pos_group'] != 'Inne']
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='pos_group', y='height_cm', data=filtered_data)
    plt.title("Wzrost zawodników  względem roli na boisku")
    plt.xlabel("Grupa pozycji")
    plt.ylabel("Wzrost (cm)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..','Plots','BoxPlot2.png'))
    plt.savefig(file_path)

def plot3(data):
    filtered_data = data[data['pos_group'] != 'Inne']
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='pos_group', y='weight_kg', data=filtered_data)
    plt.title("Waga zawodników względem roli na boisku")
    plt.xlabel("Grupa pozycji")
    plt.ylabel("Waga (kg)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'Plots', 'BoxPlot3.png'))
    plt.savefig(file_path)






def SaveAllPlots(data1):
    plot1(data1)
    plot2(data1)
    plot3(data1)
