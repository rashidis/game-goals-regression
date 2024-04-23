
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.stats as stats

if __name__ == "__main__":

    with open("./config.json", 'r') as json_file:
        config = json.load(json_file)
        
    # Load the dataset
    X = pd.read_csv('../data/X.csv')
    y = pd.read_csv('../data/y.csv')
    y["total"] = y["home_goals"] + y["away_goals"]
    y = y["total"]


    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=config['seed'])

    # Define the LightGBM parameters
    params = {
        'objective': config['objective'],
        'metric': config['metric'],
        'num_leaves': config['num_leaves'],
        'learning_rate': config['learning_rate'],
        'feature_fraction': config['feature_fraction']
    }

    # Train model on train set
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    joblib.dump(model, '../models/lgbm_model.pkl')


    # Evaluate model on test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    # Round the pred values so the predictions are integer
    y_pred = np.round(y_pred).astype(int) 

    # Plot distributions of predicted vs. actual total goals
    plt.figure(figsize=(8, 6))
    plt.hist(y_test, alpha=0.2, label='Actual')
    plt.hist(y_pred, alpha=0.2, label='Predicted')
    plt.title('Distribution of Predicted vs. Actual Total Goals')
    plt.xlabel('Total Goals')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('../results/pred_vs_actual_hist.png')

    # Plot actual vs. predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test.values.min(), y_pred.min()), max(y_test.values.max(), y_pred.max())],
            [min(y_test.values.min(), y_pred.min()), max(y_test.values.max(), y_pred.max())],
            linestyle='--', color='red')
    plt.title('Actual vs. Predicted')
    plt.xlabel('Actual Goals')
    plt.ylabel('Predicted Goals')
    plt.savefig('../results/pred_vs_actual_residuals.png')

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(model, max_num_features=10)
    plt.title('Feature Importance')
    plt.savefig('../results/feature_importance.png')