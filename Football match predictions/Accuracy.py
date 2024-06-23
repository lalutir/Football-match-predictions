import pandas as pd
import warnings

# Hide warnings
warnings.filterwarnings('ignore')

# Load the data
acc = pd.read_excel('Predictions.xlsx')

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