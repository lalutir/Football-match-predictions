import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings
import os
import datetime

# Create timestamp
timestamp = str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'))

# Hide warnings
warnings.filterwarnings('ignore')

# Check if the files are present and not open
try:
    os.rename('Statistics/Country statistics.xlsx', 'Statistics/Country statistics.xlsx')
except FileNotFoundError:
    raise FileNotFoundError('Country statistics.xlsx not found. Please make sure the file is in the correct directory.')
except PermissionError:
    raise PermissionError('Country statistics.xlsx is currently open. Please close the file before running the script.')
except Exception as e:
    raise Exception(e)

try:
    os.rename('Predictions/Predictions.xlsx', 'Predictions/Predictions.xlsx')
except FileNotFoundError:
    raise FileNotFoundError('Predictions.xlsx not found. Please make sure the file is in the correct directory.')
except PermissionError:
    raise PermissionError('Predictions.xlsx is currently open. Please close the file before running the script.')
except Exception as e:
    raise Exception(e)

# Choose the home and away team
home_team = input("Enter the home team: ")
away_team = input("Enter the away team: ")

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
directory_path = 'Statistics/'
last_file = find_last_created_file(directory_path)
rankings = pd.read_excel(last_file, sheet_name='Rankings')
teams = list(pred['Home'].unique()) + list(pred['Away'].unique())

# Check if the game has already been played
if home_team not in rankings and away_team not in teams:
    raise ValueError('Both teams have invalid names')
elif home_team not in teams:
    raise ValueError('The home team has an invalid name')
elif away_team not in teams:
    raise ValueError('The away team has an invalid name')
elif list(pred[(pred['Home'] == home_team) & (pred['Away'] == away_team) & (pred['Predicted home goals'].isna()) & (pred['Predicted away goals'].isna())].index) == []:
    raise ValueError('This game has already been predicted or does not exist')

df1 = pd.read_excel(last_file, sheet_name=home_team).drop(['Index'], axis=1)
df2 = pd.read_excel(last_file, sheet_name=away_team).drop(['Index'], axis=1)

# Ask for the amount of players in each position
to_predict_home_defenders = df1.iloc[-1]['Defenders']
to_predict_home_midfielders = df1.iloc[-1]['Midfielders']
to_predict_home_attackers = df1.iloc[-1]['Attackers']
to_predict_home_opponent_defenders = df2.iloc[-1]['Defenders']
to_predict_home_opponent_midfielders = df2.iloc[-1]['Midfielders']
to_predict_home_opponent_attackers = df2.iloc[-1]['Attackers']

# Create the dataframes
to_predict_home_dict = {'Home': 1,
                        'Defenders': to_predict_home_defenders,
                        'Midfielders': to_predict_home_midfielders,
                        'Attackers': to_predict_home_attackers,
                        'Opponent': away_team,
                        'Opponent ranking': int(rankings[rankings['country_full'] == away_team]['rank'].values[0]),
                        'Opponent ranking points': float(rankings[rankings['country_full'] == away_team]['total_points'].values[0]),
                        'Opponent defenders': to_predict_home_opponent_defenders,
                        'Opponent midfielders': to_predict_home_opponent_midfielders,
                        'Opponent attackers': to_predict_home_opponent_attackers,
                        'Goals': None,
                        'Opponent goals': None,
                        'Competitive': 1,
                        'Avg goals': df1.iloc[-6:-2]['Goals'].mean(),
                        'Avg opponent goals': df1.iloc[-6:-2]['Opponent goals'].mean()}

to_predict_away_dict = {'Home': 0,
                        'Defenders': to_predict_home_opponent_defenders,
                        'Midfielders': to_predict_home_opponent_midfielders,
                        'Attackers': to_predict_home_opponent_attackers,
                        'Opponent': home_team,
                        'Opponent ranking': int(rankings[rankings['country_full'] == home_team]['rank'].values[0]),
                        'Opponent ranking points': float(rankings[rankings['country_full'] == home_team]['total_points'].values[0]),
                        'Opponent defenders': to_predict_home_defenders,
                        'Opponent midfielders': to_predict_home_midfielders,
                        'Opponent attackers': to_predict_home_attackers,
                        'Goals': None,
                        'Opponent goals': None,
                        'Competitive': 1,
                        'Avg goals': df2.iloc[-6:-2]['Goals'].mean(),
                        'Avg opponent goals': df2.iloc[-6:-2]['Opponent goals'].mean()}

to_predict_home = pd.DataFrame(to_predict_home_dict, index=[0])
to_predict_away = pd.DataFrame(to_predict_away_dict, index=[0])

# Concatenate the dataframes
df1 = pd.concat([df1, to_predict_home]).reset_index(drop=True)
df2 = pd.concat([df2, to_predict_away]).reset_index(drop=True)

# Scale the data
scaler = StandardScaler()
col_to_scale = ['Defenders', 'Midfielders', 'Attackers', 'Opponent ranking', 'Opponent ranking points', 'Opponent defenders', 'Opponent midfielders', 'Opponent attackers', 'Avg goals', 'Avg opponent goals']

df1[col_to_scale] = scaler.fit_transform(df1[col_to_scale])
df2[col_to_scale] = scaler.fit_transform(df2[col_to_scale])

# Create the pipeline
pipeline = Pipeline([
    ('regressor', LinearRegression())
])

# Define the hyperparameters (LinearRegression doesn't have any hyperparameters)
param_grid = {
    # LinearRegression doesn't have any relevant hyperparameters for grid search
}

# Perform the grid search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Split the data
X_1 = df1.drop(['Opponent', 'Goals', 'Opponent goals'], axis=1)
y_1_1 = df1['Goals']
y_2_1 = df1['Opponent goals']

X_2 = df2.drop(['Opponent', 'Goals', 'Opponent goals'], axis=1)
y_1_2 = df2['Goals']
y_2_2 = df2['Opponent goals']

X_1_train = X_1[:-1]
y_1_1_train = y_1_1[:-1]
y_2_1_train = y_2_1[:-1]
X_1_test = X_1[-1:]

X_2_train = X_2[:-1]
y_1_2_train = y_1_2[:-1]
y_2_2_train = y_2_2[:-1]
X_2_test = X_2[-1:]

country_1_index = df1.iloc[:-1][df1['Opponent'] == away_team].index
weights_1 = np.ones(len(y_1_1_train))
weights_1[:] = 0.5
weights_1[-10:] = 1
weights_1[-4:] = 2
for i in country_1_index:
    weights_1[i] = 2.5
weights_1[-3:] = 3
weights_1[-2:] = 4
weights_1[-1] = 5

country_2_index = df2.iloc[:-1][df2['Opponent'] == away_team].index
weights_2 = np.ones(len(y_1_2_train))
weights_2[:] = 0.5
weights_2[-10:] = 1
weights_2[-4:] = 2
for i in country_2_index:
    weights_2[i] = 2.5
weights_2[-3:] = 3
weights_2[-2:] = 4
weights_2[-1] = 5

# Fit the models
grid_search.fit(X_1_train, y_1_1_train, regressor__sample_weight=weights_1)
best_model_1 = grid_search.best_estimator_
grid_search.fit(X_1_train, y_2_1_train, regressor__sample_weight=weights_1)
best_model_2 = grid_search.best_estimator_

grid_search.fit(X_2_train, y_1_2_train, regressor__sample_weight=weights_2)
best_model_3 = grid_search.best_estimator_
grid_search.fit(X_2_train, y_2_2_train, regressor__sample_weight=weights_2)
best_model_4 = grid_search.best_estimator_

# Predict the score
pred_home_goal = int(round((best_model_1.predict(X_1_train)[0] + best_model_4.predict(X_2_train)[0]) / 2, 0)) # type: ignore
pred_away_goal = int(round((best_model_2.predict(X_1_train)[0] + best_model_3.predict(X_2_train)[0]) / 2, 0)) # type: ignore

if pred_home_goal < 0:
    pred_home_goal = 0
if pred_away_goal < 0:
    pred_away_goal = 0
    
# Update the predictions
i = pred[(pred['Home'] == home_team) & (pred['Away'] == away_team) & (pred['Predicted home goals'].isna()) & (pred['Predicted away goals'].isna())].index

pred.loc[i, 'Predicted home goals'] = int(pred_home_goal)
pred.loc[i, 'Predicted away goals'] = int(pred_away_goal)

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

# Print the result
print(f'Predicted score: {home_team} {pred_home_goal} - {pred_away_goal} {away_team}')

# Save the predictions
pred.to_excel(f'Predictions/Predictions_{timestamp}.xlsx', index=False)