import pandas as pd


def position_group(pos):
    goalkeepers = ['GK']
    defenders = ['LB', 'RB', 'CB','LCB', 'RCB', 'LWB', 'RWB']
    midfielders = ['LM', 'RM', 'RCM', 'LCM', 'CM', 'CDM', 'LDM', 'RDM', 'CAM', 'LAM', 'RAM', 'LCM','RCM']
    attackers = ['ST', 'CF', 'LW', 'RW', 'LS', 'RS', 'RF', 'LF']

    if pos in defenders:
        return 'Obrońca'
    elif pos in midfielders:
        return 'Pomocnik'
    elif pos in attackers:
        return 'Napastnik'
    elif pos in goalkeepers:
        return 'bramkarz'
    else:
        return 'Inne'



def LoadData():
    data = pd.read_csv("data/players_22.csv", dtype={25: str, 108: str})
    data['pos_group'] = data['club_position'].apply(position_group)
    return data