
'''
#I wanted to look into EURO's data. It was a great tournemant again
# the brief was not set by the FA but I feel it is my duty as a patriot to perform 
# some statistical analysis in order to better understand the patterns at play 
# although this operation does not guarantee england winning the next tournemant...
# its worth a shot (on target)
'''
'''
Contents
1. Introduction
2. Fouls v Outcomes
      a. Visualisation
      b. AB Testing - T Test
3. Cards v Outcomes
      a. Visualisation
      b. AB Testing - Chi Sq
4. Winning Margin vs SOT
    a. Visualisation
    b. Simple Linear Regression
5. Final Thoughts
'''

############################
#1. INTRODUCTION
############################

'''
Welcome to the EURO Analysis piece from SolveITSeaside. 
This piece includes python code to demonstrate visualisations, AB Testing and regressions. 
The intent of this piece is to show prospective employers and clients some useful techniques within my skillset that can add value. 

In the files you can find the csv used for the insights as well as some pictures demonstrating visualisations. 

Please feel free to copy the code and repurpose it as you see fit. 

If you have any questions I would be more than happy to discuss via email,

''' 
##################################
#2. CASE STUDY: Fouls v Outcomes
##################################

#a. i. Visualisation - Comparative Bar Chart total_cards vs count of games. 
#--------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('EURO_Highlights.csv')
data = []

# Iterate through each match in the original DataFrame
for index, row in df.iterrows():
    game = f"{row['home_team']} vs {row['away_team']}"
    # Home team data
    home_team_data = {
        'Game': game,
        'Team': row['home_team'],
        'Result': 'win' if row['home_goals'] > row['away_goals'] else 'not win',
        'Fouls': row['home_fouls']
    }
    data.append(home_team_data)
    # Away team data
    away_team_data = {
        'Game': game,
        'Team': row['away_team'],
        'Result': 'win' if row['away_goals'] > row['home_goals'] else 'not win',
        'Fouls': row['away_fouls']
    }
    data.append(away_team_data)

# Create the new DataFrame
new_df = pd.DataFrame(data)

# Display the first few rows of the new DataFrame
print(new_df.head())

grouped = new_df.groupby(['Fouls', 'Result']).size().unstack(fill_value=0)

# Plot the side-by-side bar chart
grouped.plot(kind='bar', stacked=False, figsize=(10, 6))

# Add labels and title
plt.xlabel('Number of Fouls')
plt.ylabel('Frequency (Number of Games)')
plt.title('Frequency of Fouls by Match Outcome')
plt.legend(title='Match Outcome')

# Show the plot
plt.show()


'''
<><><> Considerations <><><>
its hard to compare on the side by side version as non wins have a higher frequency than wins,

??? Solution ????
Get each data point weighted as a % in order to comapre results more effectively.
'''

#a. ii. Visualisation - Implementing Changes to view Weighted Frequencies. 
#-------------------------------------------------------------------------


# Group the data by number of fouls and match result
grouped = new_df.groupby(['Fouls', 'Result']).size().unstack(fill_value=0)

# Normalize the data so each bar represents the percentage of total wins or non-wins
grouped_normalized = grouped.div(grouped.sum(axis=0), axis=1) * 100

# Plot the side-by-side bar chart with normalized data
grouped_normalized.plot(kind='bar', stacked=False, figsize=(10, 6))

# Add labels and title
plt.xlabel('Number of Fouls')
plt.ylabel('Percentage of Games (%)')
plt.title('Percentage of Fouls by Match Outcome (Normalized)')
plt.legend(title='Match Outcome')

# Show the plot
plt.show()


#b. i. Demonstrating AB Testing: Comparing Fouls Committed by Winning vs. non Winning Teams (T-Test)
#----------------------------------------------------------------------------------------------------

import pandas as pd
from scipy import stats


# Load the data
winning_fouls = new_df[new_df['Result'] == 'win']['Fouls']
not_winning_fouls = new_df[new_df['Result'] == 'not win']['Fouls']

# Perform the T-test
t_stat, p_value = stats.ttest_ind(winning_fouls, not_winning_fouls)

print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")


'''
Ho: There is no difference in average fouls committed by loosing or winning teams
Ha: There is a difference in average fouls committed by loosing or winning teams


Interpretation of T-Test Results
T-Statistic: -0.9828

on average, winning teams committed slightly fewer fouls than non-winning teams. However, the magnitude of the T-statistic is relatively small, which suggests that the difference between the two groups is not large.

P-Value: 0.3281
As p > 0.05, we fail to reject the null hypothesis. This means that the observed difference in fouls between winning and loosing teams is not statistically significant at the 5% significance level.

Conclusion:
There is not enough evidence to conclude that winning teams commit significantly fewer fouls than loosing teams. The observed difference might be due to random chance rather than a true underlying effect.
'''

'''
<><><> Considerations <><><>
In order to perform an AB test, we reduced a game with three potential outcomes down to two and did not get significant results.

??? Solution ???
Perhaps excluding draws in this context could yield more significant results on this relationship
'''

#b. ii. Demonstrating AB Testing: Comparing Fouls Committed by Winning vs. Loosing Teams (T-Test)
#------------------------------------------------------------------------------------------------

# Load the original data
df = pd.read_csv('EURO_Highlights.csv')

# Initialize a new DataFrame to store the transformed data
data = []

# Iterate through each match in the original DataFrame
for index, row in df.iterrows():
    game = f"{row['home_team']} vs {row['away_team']}"
    # Check if the match had a clear outcome (no draw)
    if row['home_goals'] != row['away_goals']:
        # Home team data
        home_team_data = {
            'Game': game,
            'Team': row['home_team'],
            'Result': 'win' if row['home_goals'] > row['away_goals'] else 'not win',
            'Fouls': row['home_fouls']
        }
        data.append(home_team_data)
        # Away team data
        away_team_data = {
            'Game': game,
            'Team': row['away_team'],
            'Result': 'win' if row['away_goals'] > row['home_goals'] else 'not win',
            'Fouls': row['away_fouls']
        }
        data.append(away_team_data)

# Create the new DataFrame
new_df_no_draws = pd.DataFrame(data)

# Display the first few rows of the new DataFrame
print(new_df_no_draws.head())

from scipy import stats

# Separate the data into winning and non-winning teams
winning_fouls = new_df_no_draws[new_df_no_draws['Result'] == 'win']['Fouls']
not_winning_fouls = new_df_no_draws[new_df_no_draws['Result'] == 'not win']['Fouls']

# Perform the T-test
t_stat, p_value = stats.ttest_ind(winning_fouls, not_winning_fouls)

print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

'''
Previous Test Results (Including Draws):
T-Statistic: -0.9828
P-Value: 0.3281
Current Test Results (Excluding Draws):
T-Statistic: -1.1757
P-Value: 0.2439

the difference between the average number of fouls committed by winning teams and non-winning teams increases slightly. Winning teams, on average, commit fewer fouls compared to non-winning teams, but this difference remains relatively small.
shows a slightly stronger (but still small) difference in fouls between winning and non-winning teams, with a p-value of 0.2439. However, this difference is still not statistically significant.
still can't reject Ho

'''

################################
#2. CASE STUDY: Cards v Outcomes
################################

'''
Rationale 
We saw in the last test that reducing the outcomes to simpply two outcomes didn't yield significant results
We now want to look at cards instead of fouls and investigate if, this time, we can get what we are looking for!
'''

#a. i. Stacked Bar Chart Total Cards vs Match Outcomes
#-----------------------------------------------------

# Load the original data
df = pd.read_csv('EURO_Highlights.csv')

# Initialize a new DataFrame to store the transformed data
data = []

# Iterate through each match in the original DataFrame
for index, row in df.iterrows():
    game = f"{row['home_team']} vs {row['away_team']}"
    # Home team data
    home_team_data = {
        'Game': game,
        'Team': row['home_team'],
        'Result': 'win' if row['home_goals'] > row['away_goals'] else ('lose' if row['home_goals'] < row['away_goals'] else 'draw'),
        'Total_Cards': row['home_total_cards']
    }
    data.append(home_team_data)
    # Away team data
    away_team_data = {
        'Game': game,
        'Team': row['away_team'],
        'Result': 'win' if row['away_goals'] > row['home_goals'] else ('lose' if row['away_goals'] < row['home_goals'] else 'draw'),
        'Total_Cards': row['away_total_cards']
    }
    data.append(away_team_data)

# Create the new DataFrame
new_df = pd.DataFrame(data)

# Group the data by the number of total cards and match outcome
grouped = new_df.groupby(['Total_Cards', 'Result']).size().unstack(fill_value=0)

# Plot the stacked bar graph
grouped.plot(kind='bar', stacked=True, figsize=(10, 6))

# Add labels and title
plt.xlabel('Total Number of Cards')
plt.ylabel('Number of Games')
plt.title('Number of Games vs. Total Cards by Match Outcome')
plt.legend(title='Match Outcome')

# Show the plot
plt.show()

'''
<><><> Consdierations <><><>
It's hard to compare with this version of the stacked chart as some have much higher frequencies.

??? Solution ???
100% stacked chart
'''

#a. ii. 100% Stacked Bar Chart Total Cards vs Match Outcomes
#-----------------------------------------------------------

# Group the data by the number of total cards and match outcome
grouped = new_df.groupby(['Total_Cards', 'Result']).size().unstack(fill_value=0)

# Normalize the data to sum to 100% for each Total_Cards
grouped_percent = grouped.div(grouped.sum(axis=1), axis=0) * 100

# Plot the 100% stacked bar graph
grouped_percent.plot(kind='bar', stacked=True, figsize=(10, 6))

# Add labels and title
plt.xlabel('Total Number of Cards')
plt.ylabel('Percentage of Games (%)')
plt.title('Percentage of Games vs. Total Cards by Match Outcome (100% Stacked)')
plt.legend(title='Match Outcome')

# Show the plot
plt.show()


#b. i. Demonstrating Categorical Data Analysis: Comparing Cards vs Outcome (Chi- SQ)
#-----------------------------------------------------------------------------------
import pandas as pd
from scipy.stats import chi2_contingency

# Load the original data
df = pd.read_csv('EURO_Highlights.csv')

# Initialize a new DataFrame to store the transformed data
data = []

# Iterate through each match in the original DataFrame
for index, row in df.iterrows():
    game = f"{row['home_team']} vs {row['away_team']}"
    # Home team data
    home_team_data = {
        'Game': game,
        'Team': row['home_team'],
        'Result': 'win' if row['home_goals'] > row['away_goals'] else ('lose' if row['home_goals'] < row['away_goals'] else 'draw'),
        'Total_Cards': row['home_total_cards']
    }
    data.append(home_team_data)
    # Away team data
    away_team_data = {
        'Game': game,
        'Team': row['away_team'],
        'Result': 'win' if row['away_goals'] > row['home_goals'] else ('lose' if row['away_goals'] < row['home_goals'] else 'draw'),
        'Total_Cards': row['away_total_cards']
    }
    data.append(away_team_data)

# Create the new DataFrame
new_df = pd.DataFrame(data)

# Group the data by the number of total cards and match outcome to create a contingency table
contingency_table = pd.crosstab(new_df['Total_Cards'], new_df['Result'])

# Perform the Chi-Square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Output the results
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(expected)

'''

Ho: There is no association between the number of cards a team receives (total cards) and the match outcome (win, lose, draw).
Ha: There is an association between the number of cards a team receives (total cards) and the match outcome (win, lose, draw).

Chi-Square Statistic: 14.2304

A higher value generally indicates a greater difference between observed and expected frequencies.

P-Value: 0.7140

This means that the observed differences between the number of cards and match outcomes (win, lose, draw) are likely due to random chance rather than a significant association.

Degrees of Freedom: 18

Fail to reject Ho

'''


'''
<><><> Considerations <><><>

It has been dificult to get a significant result across the board

??? Solutions ???

- hgiher sample size : looking at league data as there are more matches
- sub groups : categorising our data into similar groups to see if significant results exist within
- outliers : removing outliers may help product a result that accurately describes the most common behaviour

'''

#b. i. Demonstrating Categorical Data Analysis: Comparing Cards vs Outcome (Chi- SQ)
#------------------------------------------------------------------------------------

import pandas as pd
from scipy.stats import chi2_contingency

# Load the original data
df = pd.read_csv('EURO_Highlights.csv')

# Initialize a new DataFrame to store the transformed data
data = []

# Iterate through each match in the original DataFrame
for index, row in df.iterrows():
    if row['home_goals'] > row['away_goals']:  # Home team wins
        win_size = row['home_goals'] - row['away_goals']
        home_team_data = {
            'Game': f"{row['home_team']} vs {row['away_team']}",
            'Team': row['home_team'],
            'Win_Size': '+4' if win_size >= 4 else f"+{win_size}",
            'Cards_Category': '0-1' if row['home_total_cards'] <= 1 else ('2-3' if row['home_total_cards'] <= 3 else '4+')
        }
        data.append(home_team_data)
    elif row['away_goals'] > row['home_goals']:  # Away team wins
        win_size = row['away_goals'] - row['home_goals']
        away_team_data = {
            'Game': f"{row['home_team']} vs {row['away_team']}",
            'Team': row['away_team'],
            'Win_Size': '+4' if win_size >= 4 else f"+{win_size}",
            'Cards_Category': '0-1' if row['away_total_cards'] <= 1 else ('2-3' if row['away_total_cards'] <= 3 else '4+')
        }
        data.append(away_team_data)

# Create the new DataFrame
winning_df = pd.DataFrame(data)

# Create a contingency table
contingency_table = pd.crosstab(winning_df['Win_Size'], winning_df['Cards_Category'])

# Perform the Chi-Square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Output the results
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(expected)

# Display the contingency table for reference
print("Contingency Table:")
print(contingency_table)

'''
Ho: no association between win size and card categories
Ha: some association between win size and card categories

Chi-Square Statistic: 5.0462
Chi-Square statistic indicates a larger difference between the observed and expected values

P-Value: 0.5379
at a 95% confidence level fail to reject the null hypothesis

Comparison:
Chi-Square Statistic:
Previous Test: 14.2304
Current Test: 5.0462
Comparison: The Chi-Square statistic was higher in the previous test, likely because there were more degrees of freedom (more categories to compare), which allowed for more variation between observed and expected frequencies.

P-Value:
Previous Test: 0.7140
Current Test: 0.5379
Comparison: Both p-values are relatively high, indicating that neither test found significant associations. The current test has a slightly lower p-value, but it's still far from the typical significance threshold of 0.05.

Consistency in Results: 
Both tests suggest that the number of cards a team receives does not have a strong relationship with the match outcome or the size of the victory.

Potential next steps:
larger sample sizes, 
refining categories further,  
exploring other variables that might interact with win size or card count, ie. 
        team strength
        referee behavior
        player discipline strategies

'''

######################################################
#4. CASE STUDY: Team Shots on Target vs Winning Margin
######################################################

'''
Rationale 
One key aspect of performance is how efficiently a team converts 
opportunities (shots on target) into goals, ultimately influencing the margin of victory in a match.
'''

#a. i. Visualisation: Scatter Graph Total SOT vs Winning Margin
#--------------------------------------------------------------

# Load the original data
df = pd.read_csv('EURO_Highlights.csv')

# Initialize a new DataFrame to store the transformed data
data = []

# Iterate through each match in the original DataFrame
for index, row in df.iterrows():
    if row['home_goals'] > row['away_goals']:  # Home team wins
        win_size = row['home_goals'] - row['away_goals']
        home_team_data = {
            'Game': f"{row['home_team']} vs {row['away_team']}",
            'Team': row['home_team'],
            'Win_Size': win_size,
            'SOT': row['home_SOT_all'] 
        }
        data.append(home_team_data)
    elif row['away_goals'] > row['home_goals']:  # Away team wins
        win_size = row['away_goals'] - row['home_goals']
        away_team_data = {
            'Game': f"{row['home_team']} vs {row['away_team']}",
            'Team': row['away_team'],
            'Win_Size': win_size,
            'SOT': row['away_SOT_all'] 
        }
        data.append(away_team_data)

# Create the new DataFrame
winning_df = pd.DataFrame(data)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Scatter plot: Shots on Target (SOT) vs Win Margin (Win_Size)
plt.figure(figsize=(10, 6))

# Create the scatter plot
plt.scatter(winning_df['SOT'], winning_df['Win_Size'], color='blue', alpha=0.7)

# Calculate the line of best fit (trend line)
m, b = np.polyfit(winning_df['SOT'], winning_df['Win_Size'], 1)

# Plot the trend line
plt.plot(winning_df['SOT'], m*winning_df['SOT'] + b, color='red', linewidth=2)

# Add labels and title
plt.xlabel('Shots on Target (SOT)')
plt.ylabel('Win Margin (Win_Size)')
plt.title('Scatter Plot of Win Margin vs Shots on Target with Trend Line')

# Show the plot
plt.show()


#b. i. Simple Linear Regression:  Total SOT vs Winning Margin
#------------------------------------------------------------


import pandas as pd
import statsmodels.api as sm


# Define the independent variable (X) and dependent variable (Y)
X = winning_df['SOT']
Y = winning_df['Win_Size']

# Add a constant to the independent variable (required for statsmodels)
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(Y, X).fit()

# Print the summary of the regression
print(model.summary())

'''
   
Ho: There is no relationship between Shot on Target and Winning Margins
Ha: There is a relationship between Shot on Target and Winning Margins
                         OLS Regression Results                            
==============================================================================
Dep. Variable:               Win_Size   R-squared:                       0.217
Model:                            OLS   Adj. R-squared:                  0.192
Method:                 Least Squares   F-statistic:                     8.857
Date:                Tue, 13 Aug 2024   Prob (F-statistic):            0.00552
Time:                        18:29:39   Log-Likelihood:                -38.190
No. Observations:                  34   AIC:                             80.38
Df Residuals:                      32   BIC:                             83.43
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.7872      0.308      2.552      0.016       0.159       1.416
SOT            0.1510      0.051      2.976      0.006       0.048       0.254
==============================================================================
Omnibus:                        7.614   Durbin-Watson:                   1.814
Prob(Omnibus):                  0.022   Jarque-Bera (JB):                6.742
Skew:                           1.082   Prob(JB):                       0.0343
Kurtosis:                       3.283   Cond. No.                         14.6
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

R2: of 0.217 suggests that shots on target have some predictive power, it also indicates that other factors likely contribute to the win margin.


Prob (F-Statistic) (0.00552): 
This indicates that the model as a whole is statistically significant and that shots on target (SOT) are a significant predictor of win margin.

Intercept (const): 0.7872
p value: 0.016, 
statistically significant at the 0.05 level.

SOT Coefficient: 0.1510
p value: 0.006,
statistically significant suggesting a meaningful relationship
For every 1-unit increase in Shots on Target (SOT) by the winning team, the win margin (Win_Size) is expected to increase by 0.151 goals

Omnibus (7.614) and Prob (Omnibus) (0.022): 
A low p-value (as seen here) suggests that the residuals may deviate from a normal distribution, 
which could indicate some issues with the model fit or the presence of outliers.

Durbin-Watson (1.814): 
A value close to 2 suggests that there is little to no autocorrelation, which is desirable.

'''

#b. ii. Visualising the Regression
#----------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


# Define the independent variable (X) and dependent variable (Y)
X = winning_df['SOT']
Y = winning_df['Win_Size']

# Add a constant to the independent variable (required for statsmodels)
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(Y, X).fit()

# Scatter plot: Shots on Target (SOT) vs Win Margin (Win_Size)
plt.figure(figsize=(10, 6))

# Create the scatter plot
plt.scatter(winning_df['SOT'], winning_df['Win_Size'], color='blue', alpha=0.7)

# Calculate the line of best fit (trend line)
m, b = np.polyfit(winning_df['SOT'], winning_df['Win_Size'], 1)

# Plot the trend line
plt.plot(winning_df['SOT'], m*winning_df['SOT'] + b, color='red', linewidth=2)

# Add labels and title
plt.xlabel('Shots on Target (SOT)')
plt.ylabel('Win Margin (Win_Size)')
plt.title('Regression Line of Win Margin vs Shots on Target')

# Show the plot
plt.show()

'''
Reflections


The model suggests that teams should focus on increasing their shots on target 
to improve their chances of winning by larger margins. However, the relatively modest 
effect size implies that teams should also consider other areas of performance.

While the model is statistically significant, its explanatory power is modest, indicating 
that many other factors contribute to a team's success

Moving forward, incorporating additional variables and considering the potential 
non-linearity of relationships will be key to developing a more robust predictive model. 
This analysis should be viewed as an initial step in a more comprehensive examination of 
football performance metrics
'''


##################
#5. FINAL THOUGHTS
##################

'''
Final Thoughts
This project provided a valuable exploration of statistical methods applied to football data, 
specifically from the EUROs tournament. Through various analyses, including T-tests, Chi-Square tests, 
and a linear regression, we sought to uncover meaningful patterns that might influence match outcomes 
and win margins. While the analyses yielded some insights, such as a modest positive relationship between 
shots on target and win margin, they also highlight the complexity of football performance and the limitations 
of using single variables to predict outcomes.

One of the key takeaways is the importance of context and the multifaceted nature of football matches. 
The relatively low explanatory power of the regression model, for example, suggests that while shots on target 
are important, they are far from the only factor determining how much a team wins by. Similarly, the lack of 
significant results in the Chi-Square tests points to the need for larger sample sizes or more refined models that 
can better capture the intricacies of match dynamics.

Looking ahead, future analyses would benefit from incorporating a broader range of variables, such as team strength, 
player-specific data, and game context factors. Additionally, exploring non-linear models or machine learning techniques 
could provide deeper insights and more accurate predictions. This project serves as a solid foundation for understanding 
some basic patterns, but it also underscores the potential for more sophisticated analyses to drive actionable insights 
in football analytics.

'''