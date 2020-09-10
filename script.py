'''
Project Goals

In this project I created linear regression models that include single, two feature and multiple that predicts the outcomes
for a tennis player based on their playing habits.
By analyzing and modeling the Association of Tennis Professionals (ATP) data.

'''
import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# load and investigate the data here:
df = pd.read_csv('tennis_stats.csv')
print(df.head())
print(df.columns)
print(df.describe())
# perform exploratory analysis here:

plt.scatter(df['BreakPointsOpportunities'],df['Winnings'])
plt.title('BreakPointsOpportunities vs Winnings')
plt.xlabel('BreakPointsOpportunities')
plt.ylabel('Winnings')
plt.show()
plt.clf()

plt.scatter(df['BreakPointsFaced'],df['Wins'])
plt.title('BreakPointsFaced vs Wins')
plt.xlabel('BreakPointsFaced')
plt.ylabel('Wins')
plt.show()
plt.clf()

plt.scatter(df['ServiceGamesPlayed'],df['Wins'])
plt.title('ServiceGamesPlayed vs Wins')
plt.xlabel('ServiceGamesPlayed')
plt.ylabel('Wins')
plt.show()
plt.clf()

plt.scatter(df['ServiceGamesPlayed'],df['Winnings'])
plt.title('ServiceGamesPlayed vs Winnings')
plt.xlabel('ServiceGamesPlayed')
plt.ylabel('Winnings')
plt.show()
plt.clf()


plt.scatter(df['FirstServePointsWon'],df['Ranking'])
plt.title('FirstServePointsWon vs Ranking')
plt.xlabel('FirstServePointsWon')
plt.ylabel('Ranking')
plt.show()
plt.clf()

## Perform single feature linear regressions here: Service Games Played

x = df[['ServiceGamesPlayed']]
y = df[['Wins']]

# train, test, split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, test_size = 0.2)

#create linear regression model then fit data

lrm = LinearRegression()
lrm.fit(x_train,y_train)

# check prediction score 
print('Predicted Wins from Service Games Played: ', lrm.score(x_test,y_test))

# predictive model 
y_predict = lrm.predict(x_test)

# scatter plot
plt.scatter(x_test, y_predict, alpha = 0.4)
plt.title('Predicted Wins vs. Actual Wins')
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.show()
plt.clf()

## Perform single feature linear regressions here: Break Points Opportunities

x = df[['BreakPointsOpportunities']]
y = df[['Winnings']]

# train, test, split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, test_size = 0.2)

#create linear regression model then fit data

lrm = LinearRegression()
lrm.fit(x_train,y_train)

# check prediction score 
print('Predicted Winnings from Break Points Opportunities: ', lrm.score(x_test,y_test))

# predictive model 
y_predict = lrm.predict(x_test)

# scatter plot
plt.scatter(x_test, y_predict, alpha = 0.4)
plt.title('Predicted Winnings vs. Actual Winnings')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()

# Performing two stat Linear Regression Break Points Opportunities, FirstServeReturnPointsWon
# select features and value to predict

x = df[['BreakPointsOpportunities','FirstServeReturnPointsWon']]
y = df[['Winnings']]

# train, test, split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)

# create and train model on training data
lrm = LinearRegression()
lrm.fit(x_train,y_train)

# score model on test data
print('Predicting Winnings with 2 Features Test Score:', lrm.score(x_test,y_test))

# make predictions with model
y_predict = lrm.predict(x_test)

# plot predictions against actual winnings
plt.scatter(y_test,y_predict, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - 2 Features')
plt.xlabel('Actual Winnings: Break Points Opportunities + FirstServeReturnPointsWon')
plt.ylabel('Predicted Winnings: Break Points Opportunities + FirstServeReturnPointsWon')
plt.show()
plt.clf()

# Performing two stat Linear Regression Service Games Played + Break Points Faced, Winnings

# select features and value to predict
x = df[['ServiceGamesPlayed','BreakPointsFaced']]
y = df[['Winnings']]

# train, test, split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)

# create and train model on training data
lrm = LinearRegression()
lrm.fit(x_train,y_train)

# score model on test data
print('Predicting Winnings with 2 Features Test Score:  ', lrm.score(x_test,y_test))

# make predictions with model
y_predict = lrm.predict(x_test)

# plot predictions against actual winnings
plt.scatter(y_test,y_predict, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - 2 Features')
plt.xlabel('Actual Winnings: ServiceGamesPlayed + BreakPointsFaced')
plt.ylabel('Predicted Winnings: ServiceGamesPlayed + BreakPointsFaced')
plt.show()
plt.clf()

# Multiple Linear Regression to Predict Yearly Earnings.


x = df[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon','SecondServePointsWon','SecondServeReturnPointsWon','Aces','BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities','BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon','ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon','TotalServicePointsWon']]

y = df[['Winnings']]

# train, test, split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)

# create and train model on training data
lrm = LinearRegression()
lrm.fit(x_train,y_train)

# score model on test data
print('Predicting Winnings with Multiple Features Test Score:', lrm.score(x_test,y_test))

# make predictions with model
y_predict = lrm.predict(x_test)

# plot predictions against actual winnings
plt.scatter(y_test,y_predict, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - Mulitiple Features')
plt.xlabel('Actual Winnings:')
plt.ylabel('Predicted Winnings:')
plt.show()
plt.clf()

