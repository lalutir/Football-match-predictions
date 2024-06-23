import pandas as pd
import warnings
import datetime
import os

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

# Load the data
directory_path = 'Predictions/'
last_file = find_last_created_file(directory_path)
acc = pd.read_excel(last_file)

# Filter for already played games
acc = acc[acc['Actual home goals'].notna()]

# Sum the correct predictions
correct_score = acc['Correct score'].sum()
correct_toto = acc['Correct TOTO'].sum()
correct_overall = acc['Correct overall'].sum()
total = len(acc)

# Create the dataframe
dict = {'Correct': ['Score/overall', 'TOTO'],
        'Correct predictions': [correct_score, correct_toto],
        'Total': [total, total],
        'Percentage': [round(correct_score / total, 2), round(correct_toto / total, 2)]}

acc = pd.DataFrame(dict)

# Display the dataframe
print(acc)

acc.to_csv(f'Accuracy/Accuracy_{timestamp}.csv', index=False)