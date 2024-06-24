import pandas as pd
import warnings
import os
import datetime

# Create timestamp
timestamp = str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'))

# Hide warnings
warnings.filterwarnings('ignore')

# Check if the files are present and not open
try:
    os.rename('Predictions/Predictions.xlsx', 'Predictions/Predictions.xlsx')
except FileNotFoundError:
    raise FileNotFoundError('Predictions.xlsx not found. Please make sure the file is in the correct directory.')
except PermissionError:
    raise PermissionError('Predictions.xlsx is currently open. Please close the file before running the script.')
except Exception as e:
    raise Exception(e)

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

# Load the data
directory_path = 'Predictions/'
last_file = find_last_created_file(directory_path)
pred = pd.read_excel(last_file)

# Choose the home and away team
home_team = input("Enter the home team: ")
away_team = input("Enter the away team: ")
teams = list(pred['Home'].unique()) + list(pred['Away'].unique())

# Check if the game has already been played
if home_team not in teams and away_team not in teams:
    raise ValueError('Both teams have invalid names')
elif home_team not in teams:
    raise ValueError('The home team has an invalid name')
elif away_team not in teams:
    raise ValueError('The away team has an invalid name')
elif list(pred[(pred['Home'] == home_team) & (pred['Away'] == away_team) & (pred['Actual home goals'].isna()) & (pred['Actual away goals'].isna())].index) == []:
    raise ValueError('This game has already been played or does not exist')

# Input score
home_goal = int(input('Home goals: '))
away_goal = int(input('Away goals: '))

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