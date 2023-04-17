import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error





def linear_regression(df):
    
    y = df.reb
    
    features = ['player_height', 'player_weight']
    feature = ['player_height']
    
    X = df[features]
    
    # print(X)
    
    midpoint = X.shape[0]//2
    X_train = X[:midpoint]
    X_test = X[midpoint:]
    
    y_train = y[:midpoint]
    y_test = y[midpoint:]
    
    
    
    
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    print("Coefficients are: ", reg.coef_)
    
    ax = plt.axes()
    # ax.scatter(X_test, y_test, color = 'black')
    ax.plot(X_test.iloc[:,0], y_pred, color="blue")
    plt.show()


def draft_v_undraft(df):
    
    undrafted_players = df[df['draft_year'] == 'Undrafted']
    drafted_players = df[df['draft_year'] != 'Undrafted']
    
    num_pts_above_20_undrafted = undrafted_players[undrafted_players['pts'] > 20].shape[0]
    num_pts_above_20_drafted = drafted_players[drafted_players['pts'] > 20].shape[0]
    print(undrafted_players[undrafted_players['pts'] > 20])
    # print("The number of drafted players who have averaged above 20 pts is ", num_pts_above_20_drafted, " vs the number of undrafted players who averaged above 20 ", num_pts_above_20_undrafted)



def sort_by_decade(df):
    
    df = df.drop(df[df['draft_round'] == 'Undrafted'].index)
    df['draft_year'] = df['draft_year'].map(int)
    # df.loc[df['draft_year'].between(1970, 1980, 'left'), 'decade'] = '1970s'
    # df.loc[df['draft_year'].between(1980, 1990, 'left'), 'decade'] = '1980s'
    # df.loc[df['draft_year'].between(1990, 2000, 'left'), 'decade'] = '1990s'
    # df.loc[df['draft_year'].between(2000, 2010, 'left'), 'decade'] = '2000s'
    # df.loc[df['draft_year'].between(2010, 2020, 'left'), 'decade'] = '2010s'
    
    bins = [1970,1980,1990,2000,2010,2020, 2030]
    labels = ['1970s', '1980s', '1990s', '2000s','2010s', '2020s']
    df['decade'] = pd.cut(x = df['draft_year'], bins = bins, right=False, labels=labels, include_lowest=False)
    
    return df

def classify_tree(df):
    # print(df)
    features = ['player_name', 'draft_year', 'net_rating']
    X_orig = df[features]
    X = X_orig.iloc[:,1:]
    y = df['pts']
    
    nba_model = DecisionTreeRegressor(random_state=1)
    nba_model.fit(X,y)
    
    y_pred = nba_model.predict(X)
    mean_abs_error = mean_absolute_error(y, y_pred)
    
    print(mean_abs_error)
    

def main():
    
    df = pd.read_csv(r"C:\Users\tasoglum\Desktop\archive\all_seasons.csv").iloc[:,1:]
    
    # df = df.drop(df[df['draft_year'] == 'Undrafted'].index, axis=1)
    # draft_v_undraft(df)
    df = sort_by_decade(df)
    # print(df)
    classify_tree(df)
    
    # Want to predict total number of rebounds based on height
    
    # The target is rebounds
    
        
    return 0





if __name__ == "__main__":
    main()