import statsmodels.api as sm
from scipy.stats import pointbiserialr
import pandas as pd
import seaborn as sns
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


df = pd.read_csv("data.csv")
df = df.dropna()
df = df.drop(['Date', 'Year'], axis=1)

print(df['PitchType'].value_counts()['CHANGEUP'])
print(df['PitchType'].value_counts()['FASTBALL'])
print(df['PitchType'].value_counts()['CUTTER'])
print(df['PitchType'].value_counts()['CURVEBALL'])

df = df.drop(['Balls', 'Inning', 'PAofInning', 'Strikes', 'PitchofPA',
             'Pitcher', 'PitcherThrows', 'whiff_prob_gs'], axis=1)
df = pd.get_dummies(df)


# Correlational analysis for the feature variables
# col1 = df['whiff_prob']
# col2 = df['PlateSide']
# col3 = df['PlateHeight']
# col4 = df['PitchType_CHANGEUP']

# data = pd.DataFrame({'Whiff_Prob': col1, 'PlateSide': col2,
#                     'PlateHeight': col3, 'ChangeUp': col4})

# corr = data.corr()

# sns.heatmap(corr,
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values)

# plt.show()


# Divide the data into subsets for left and right handed batters to calculate correlations
# subsetLefty = df[df['BatterSide'] == 'Left']
# subsetRighty = df[df['BatterSide'] == 'Right']

# # Select the columns containing the variables
# col1Left = subsetLefty['whiff_prob']
# col2Left = subsetLefty['PlateSide']

# col1Right = subsetRighty['whiff_prob']
# col2Right = subsetRighty['PlateSide']

# # Calculate the correlation coefficient
# corrLeft = col1Left.corr(col2Left)
# corrRight = col1Right.corr(col2Right)

# print(corrLeft)
# print(corrRight)

# FINDING THE IMPORTANT FEATURES
# ------------------------------

# Select the independent variables and the dependent variable
X = df.drop(columns=['whiff_prob', "swing_prob", "SpinAxis"])
y = df['whiff_prob']

print(X.columns)


# Create a RandomForestRegressor model
model = RandomForestRegressor()

# Fit the model

model.fit(X, y)

# Get the feature importances, using our model
importances = model.feature_importances_

# Sort the features by their importance
sorted_indices = np.argsort(importances)[::-1]

# Select the top N features
N = 5
selected_features = [X.columns[i] for i in sorted_indices[:N]]

# Create a bar plot of the feature importances
plt.bar(range(len(X.columns)), importances)
plt.xticks(range(len(X.columns)), X.columns, rotation=90)
plt.xlabel('Variable')
plt.ylabel('Importance')
plt.title('Variable Importance')
plt.show()


# RUNNING THE CORRELATIONAL FEATURES
# ------------------------------

corrMatrix = pd.DataFrame(df.corr()).abs()
corrMatrix.loc['average'] = corrMatrix.mean()
sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
plt.show()


# BUILDING AND TRAINING THE MODEL
# ------------------------------

# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Pick and use a model, I used a GradientBoostingRegressor
model2 = GradientBoostingRegressor(
    loss='ls', learning_rate=0.1, n_estimators=100)

# Fit the model, find predicted values
model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)


# Calculate useful statistics to evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# CREATE PLOTS TO EVALUATE MODEL
# ------------------------------

# Create a scatter plot of the predicted values (y_pred) and the true values (y_test)
plt.scatter(y_test, y_pred)

# Add a line of best fit
plt.plot([y_test.min(), y_test.max()], [
         y_test.min(), y_test.max()], 'k--', lw=2)

# Add labels and title
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Predicted vs. true values')

# Show the plot
plt.show()

# Calculate the residuals
residuals = y_test - y_pred

# Create a histogram of the residuals
plt.hist(residuals, bins=50)

# Add a vertical line at the mean of the residuals
plt.axvline(residuals.mean(), color='r', linestyle='dashed', linewidth=2)

# Add labels and title
plt.xlabel('Residual')
plt.ylabel('Count')
plt.title('Histogram of residuals')

# Show the plot
plt.show()

# ADJUSTING THE MODEL TO FIND A SWEET SPOT
# ----------------------------------------

# First, segment the data into Lefties and Righties to see the impact on each type of batter:
X_test_right = []
X_test_left = []

# Then, make a note of the whiff probability prediction prior to adjustments
beforeChange = sum(model2.predict(X_test)) / len(model2.predict(X_test))
print('Before Change : ', beforeChange)

# Create a copy of the data, and then adjust the PlateHeight both up and down to see the impacts
X_test_copy1 = X_test.copy()

# Shift the PlateHeight down by half to see if we can artificially increase whiff probability rates
X_test_copy1['PlateHeight'] = X_test_copy1['PlateHeight'] - \
    (X_test_copy1['PlateHeight'] / 2)
afterChangePlateHeightDown = sum(
    model2.predict(X_test_copy1)) / len(X_test_copy1)
print('PlateHeight Down New : ', afterChangePlateHeightDown)

# Shift the PlateHeight up by half to see if we can artificially increase whiff probability rates
X_test_copy1['PlateHeight'] = X_test_copy1['PlateHeight'] + \
    (X_test_copy1['PlateHeight'] / 2)
afterChangePlateHeightUp = sum(
    model2.predict(X_test_copy1)) / len(X_test_copy1)
print('PlateHeight Up New : ', afterChangePlateHeightUp)

# Create another copy of the data, and then adjust the PlateSide both up and down to see the impacts
X_test_copy1 = X_test.copy()

# Shift the PlateSide left by half to see if we can artificially increase whiff probability rates
X_test_copy1['PlateSide'] = X_test_copy1['PlateSide'] - \
    (X_test_copy1['PlateSide'] / 2)
afterChangePlateSideLeft = sum(
    model2.predict(X_test_copy1)) / len(X_test_copy1)
print('PlateSide Left New : ', afterChangePlateSideLeft)

# Shift the PlateSide right by half to see if we can artificially increase whiff probability rates
X_test_copy1['PlateSide'] = X_test_copy1['PlateSide'] + \
    (X_test_copy1['PlateSide'] / 2)
afterChangePlateSideRight = sum(
    model2.predict(X_test_copy1)) / len(X_test_copy1)
print('PlateSide Right New : ', afterChangePlateSideRight)

# Create another copy of the data, and then adjust the PitchType to replace Curveballs/Fastballs/Cutters with Changeups
X_test_copy1 = X_test.copy()

# Replace Fastballs with Changeups to see the impact on our model
X_test_copy1.loc[X_test_copy1['PitchType_FASTBALL']
                 > 0, 'PitchType_CHANGEUP'] = 1
X_test_copy1.loc[X_test_copy1['PitchType_FASTBALL']
                 > 0, 'PitchType_FASTBALL'] = 0
afterChangeMoreChangeLessFastball = sum(
    model2.predict(X_test_copy1)) / len(X_test_copy1)
print('Replace Fastballs w/Changeups New : ',
      afterChangeMoreChangeLessFastball)

X_test_copy1 = X_test.copy()


# Replace Curveballs with Changeups to see the impact on our model
X_test_copy1.loc[X_test_copy1['PitchType_CURVEBALL']
                 > 0, 'PitchType_CHANGEUP'] = 1
X_test_copy1.loc[X_test_copy1['PitchType_CURVEBALL']
                 > 0, 'PitchType_CURVEBALL'] = 0
afterChangeMoreChangeLessCurve = sum(
    model2.predict(X_test_copy1)) / len(X_test_copy1)
print('Replace Curveballs w/Changeups New : ', afterChangeMoreChangeLessCurve)


X_test_copy1 = X_test.copy()


# Replace Cutters with Changeups to see the impact on our model
X_test_copy1.loc[X_test_copy1['PitchType_CUTTER']
                 > 0, 'PitchType_CHANGEUP'] = 1
X_test_copy1.loc[X_test_copy1['PitchType_CUTTER']
                 > 0, 'PitchType_CUTTER'] = 0
afterChangeMoreChangeLessCutter = sum(
    model2.predict(X_test_copy1)) / len(X_test_copy1)
print('Replace Cutters w/Changeups New : ', afterChangeMoreChangeLessCutter)


# NEXT STEP - COMBINE CHANGES TO BUILD A STRONGER MODEL

# Try doing it all at once and turning everything into a Changeup
# X_test_copy3.loc[X_test_copy3['PitchType_FASTBALL']
#                  > 0, 'PitchType_CHANGEUP'] = 1
# X_test_copy3.loc[X_test_copy3['PitchType_FASTBALL']
#                  > 0, 'PitchType_FASTBALL'] = 0
# X_test_copy3.loc[X_test_copy3['PitchType_CUTTER']
#                  > 0, 'PitchType_CHANGEUP'] = 1
# X_test_copy3.loc[X_test_copy3['PitchType_CUTTER']
#                  > 0, 'PitchType_CUTTER'] = 0
# X_test_copy3.loc[X_test_copy3['PitchType_CURVEBALL']
#                  > 0, 'PitchType_CHANGEUP'] = 1
# X_test_copy3.loc[X_test_copy3['PitchType_CURVEBALL']
#                  > 0, 'PitchType_CURVEBALL'] = 0
# changeEverythingIntoChangeup = sum(
#     model2.predict(X_test_copy3)) / len(X_test_copy3)
# print('Turn everything into Changeups - New : ',
#       changeEverythingIntoChangeup)


# Based on current findings, it seems ideal to lower the height, throw a changeup, and pitch more to the corners of the plate
# Run this test on both righties and lefties

# Righty

X_test_copy1 = X_test.copy()
X_test_copy1['BatterSide_Left'] = 0
X_test_copy1['BatterSide_Right'] = 1
X_test_copy1['PlateHeight'] = X_test_copy1['PlateHeight'] - \
    (X_test_copy1['PlateHeight'] / 2)
X_test_copy1['PlateSide'] = X_test_copy1['PlateSide'] - \
    (X_test_copy1['PlateSide'] / 2)
X_test_copy1.loc[X_test_copy1['PitchType_FASTBALL']
                 > 0, 'PitchType_CHANGEUP'] = 1
X_test_copy1.loc[X_test_copy1['PitchType_FASTBALL']
                 > 0, 'PitchType_FASTBALL'] = 0

afterChangePlateSideLeftAndUpAndChangeup = sum(
    model2.predict(X_test_copy1)) / len(X_test_copy1)
print('After moving Ball Down, Left, and into a Changeup to a Righty : ',
      afterChangePlateSideLeftAndUpAndChangeup)

# Lefty

X_test_copy1 = X_test.copy()
X_test_copy1['BatterSide_Left'] = 1
X_test_copy1['BatterSide_Right'] = 0
X_test_copy1['PlateHeight'] = X_test_copy1['PlateHeight'] - \
    (X_test_copy1['PlateHeight'] / 2)
X_test_copy1['PlateSide'] = X_test_copy1['PlateSide'] - \
    (X_test_copy1['PlateSide'] / 2)
X_test_copy1.loc[X_test_copy1['PitchType_FASTBALL']
                 > 0, 'PitchType_CHANGEUP'] = 1
X_test_copy1.loc[X_test_copy1['PitchType_FASTBALL']
                 > 0, 'PitchType_FASTBALL'] = 0

afterChangePlateSideLeftAndUpAndChangeup = sum(
    model2.predict(X_test_copy1)) / len(X_test_copy1)
print('After moving Ball Down, Left, and into a Changeup to a Lefty : ',
      afterChangePlateSideLeftAndUpAndChangeup)


def get_fields(schedule):
    """

    Q3 -- 

    If we reduce this problem to a Gale-Shapley instance, it is simple.
    From there, we can pretend that the Fields are the proposers and the Teams are the proposees


    Basically Gale-Shapley?

    Initialize their preferences:
        - Teams will have preferences in the order of their schedules by time slot
        - Fields will have preferences in the reverse order from their schedules by time slot

    While there is a field f who is free and hasn't yet asked every team to play:

            - Let t be the highest-ranked team in f's preference list
            to whom f has not yet asked to play

            - If t is free:
                - (t, f) decide to match up
            - Else that must mean t is currently paired up with f'
                - If t prefers f' to f then f remains free
                - Else t prefers f to f' and (t, f) become paired and f' becomes free

    Return the set of matched pairs


    -- The run-time of this algorithm is O(n^2):

        There will be at most n^2 iterations throughout this algorithm, per 1.3 in the textbook.
        This is better than O(mn) since m>n 

    -- We will prove correctness with a proof by contradiction:

        If we assume an instability exists where:

        Field 1 is paired with team 1, and field 2 is paired with team 2

        However: Field 1 prefers team 2 to team 1 and team 2 prefers field 1 to field 2

        In the execution of the algorithm that produced our final pairings, we know
        that by definition field one's last proposal was to team 1.
        We can consider if field one did ask team 2 at some point earlier if they wanted 
        to be a pair. If it didn't, then team 1 must occur higher on field one's preference list 
        than team 2, which contradicts our assumption that field one prefers team 2 to team 1.
        If it did, then it was rejected by team 2 in favor of some other team, say team 3, 
        who team 2 prefers to field 1. Field 2 is the final partner of team 2, so either
        field 3 = field 2 or by the Gale Shapley Algorithm, team 2 prefers the final
        partner field 2 to field. This does not really matter, since either way this contradicts 
        our assumption that team 2 prefers field one to field 2.

    """
