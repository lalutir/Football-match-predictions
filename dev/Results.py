import pandas as pd
import warnings
import os
import datetime

# Create timestamp
timestamp = str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'))

# Hide warnings
warnings.filterwarnings('ignore')

def find_last_created_file(directory):
    try:
        files = os.listdir(directory)
    except OSError:
        print(f"Error: Could not access directory '{directory}'")
        return None
    
    if not files:
        print(f"Directory '{directory}' is empty")
        return None
    
    files = [os.path.join(directory, file) for file in files]
    files.sort(key=os.path.getctime)
    last_file = files[-1]
    
    return last_file

# Choose the home and away team
home_team = input("Enter the home team: ")
away_team = input("Enter the away team: ")

# Load the data
directory_path = 'Predictions/'
last_file = find_last_created_file(directory_path)
pred = pd.read_excel(last_file)
rankings = pd.read_csv('Statistics/National Teams/Rankings.csv')
nt = list(rankings['country_full'].unique())
if home_team in nt and away_team in nt:
    home_team_df = pd.read_csv(f'Statistics/National Teams/{home_team}.csv')
    away_team_df = pd.read_csv(f'Statistics/National Teams/{away_team}.csv')

# Check if the game has already been played
teams = list(pred['Home'].unique()) + list(pred['Away'].unique())
if home_team not in teams and away_team not in teams:
    raise ValueError('Both teams have invalid names')
elif home_team not in teams:
    raise ValueError('The home team has an invalid name')
elif away_team not in teams:
    raise ValueError('The away team has an invalid name')
elif list(pred[(pred['Home'] == home_team) & (pred['Away'] == away_team) & (pred['Actual home goals'].isna()) & (pred['Actual away goals'].isna())].index) == []:
    raise ValueError('This game has already been played or does not exist')

# Input score
home_defenders = int(input('Home defenders: '))
home_midfielders = int(input('Home midfielders: '))
home_attackers = int(input('Home attackers: '))
home_opponent_defenders = int(input('Away defenders: '))
home_opponent_midfielders = int(input('Away midfielders: '))
home_opponent_attackers = int(input('Away attackers: '))
home_goal = int(input('Home goals: '))
away_goal = int(input('Away goals: '))
home_competitive = int(input('Competitive: '))

# Create the dataframes
home_dict = {'Home': 1,
            'Defenders': home_defenders,
            'Midfielders': home_midfielders,
            'Attackers': home_attackers,
            'Opponent': away_team,
            'Opponent ranking': int(rankings[rankings['country_full'] == away_team]['rank'].values[0]),
            'Opponent ranking points': float(rankings[rankings['country_full'] == away_team]['total_points'].values[0]),
            'Opponent defenders': home_opponent_defenders,
            'Opponent midfielders': home_opponent_midfielders,
            'Opponent attackers': home_opponent_attackers,
            'Goals': home_goal,
            'Opponent goals': away_goal,
            'Competitive': home_competitive,
            'Avg goals': home_team_df.iloc[-6:-1]['Goals'].mean(),
            'Avg opponent goals': home_team_df.iloc[-6:-1]['Opponent goals'].mean()}

away_dict = {'Home': 0,
            'Defenders': home_opponent_defenders,
            'Midfielders': home_opponent_midfielders,
            'Attackers': home_opponent_attackers,
            'Opponent': home_team,
            'Opponent ranking': int(rankings[rankings['country_full'] == home_team]['rank'].values[0]),
            'Opponent ranking points': float(rankings[rankings['country_full'] == home_team]['total_points'].values[0]),
            'Opponent defenders': home_defenders,
            'Opponent midfielders': home_midfielders,
            'Opponent attackers': home_attackers,
            'Goals': away_goal,
            'Opponent goals': home_goal,
            'Competitive': home_competitive,
            'Avg goals': away_team_df.iloc[-6:-1]['Goals'].mean(),
            'Avg opponent goals': away_team_df.iloc[-6:-1]['Opponent goals'].mean()}

home_team_add = pd.DataFrame(home_dict, index=[0])
home_team_df = pd.concat([home_team_df, home_team_add]).reset_index(drop=True)
away_team_add = pd.DataFrame(away_dict, index=[0])
away_team_df = pd.concat([away_team_df, away_team_add]).reset_index(drop=True)

home_team_df.to_csv(f'Statistics/National Teams/{home_team}.csv', index=False)
away_team_df.to_csv(f'Statistics/National Teams/{away_team}.csv', index=False)

# Update the predictions
i = pred[(pred['Home'] == home_team) & (pred['Away'] == away_team) & (pred['Actual home goals'].isna()) & (pred['Actual away goals'].isna())].index
pred.loc[i, 'Actual home goals'] = int(home_goal)
pred.loc[i, 'Actual away goals'] = int(away_goal)

# Check if the predictions are correct
pred_toto = []
act_toto = []
cor_score = []
cor_toto = []

for i in range(len(pred)):
    if pred['Predicted home goals'][i] > pred['Predicted away goals'][i]:
        pred_toto.append(1)
    elif pred['Predicted home goals'][i] < pred['Predicted away goals'][i]:
        pred_toto.append(-1)
    else:
        pred_toto.append(0)

    if pred['Actual home goals'][i] > pred['Actual away goals'][i]:
        act_toto.append(1)
    elif pred['Actual home goals'][i] < pred['Actual away goals'][i]:
        act_toto.append(-1)
    else:
        act_toto.append(0)

    if pred['Predicted home goals'][i] == pred['Actual home goals'][i] and pred['Predicted away goals'][i] == pred['Actual away goals'][i]:
        cor_score.append(1)
    else:
        cor_score.append(0)
        
    if pred_toto[i] == act_toto[i]:
        cor_toto.append(1)
    else:
        cor_toto.append(0)
        
# Add the results to the dataframe
pred['Predicted TOTO'] = pred_toto
pred['Actual TOTO'] = act_toto
pred['Correct score'] = cor_score
pred['Correct TOTO'] = cor_toto

# Save the scores
pred.to_excel(f'Predictions/Predictions_{timestamp}.xlsx', index=False)