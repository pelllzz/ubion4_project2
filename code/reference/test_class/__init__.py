import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, GridSearchCV

class ubion():
    # Ridge, Lasso, ElasticNet의 MSE, RMSE 스코어 계산하는 함수
    def score_checker(type, data, target, alpha):
        if type == 'ridge':
            type = Ridge(alpha)
            mse_score = cross_val_score(type, data, target, scoring="neg_mean_squared_error", cv=5)
            rmse_score = np.sqrt(-1 * mse_score)
            avg_rmse = np.mean(rmse_score)

            print(f'Alpha : {alpha}')
            print(f'Ridge Negative MSE score : {np.abs(np.round(mse_score, 3))}')
            print(f'Ridge RMSE scores : {np.round(rmse_score, 3)}')
            print(f'Ridge AVG RMSE : {avg_rmse:.3f}\n')

        elif type == 'lasso':
            type = Lasso(alpha)
            mse_score = cross_val_score(type, data, target, scoring="neg_mean_squared_error", cv=5)
            rmse_score = np.sqrt(-1 * mse_score)
            avg_rmse = np.mean(rmse_score)

            print(f'Alpha : {alpha}')
            print(f'Lasso Negative MSE score : {np.abs(np.round(mse_score, 3))}')
            print(f'Lasso RMSE scores : {np.round(rmse_score, 3)}')
            print(f'Lasso AVG RMSE : {avg_rmse:.3f}\n')

        elif type == 'elastic':
            type = ElasticNet(alpha)
            mse_score = cross_val_score(type, data, target, scoring="neg_mean_squared_error", cv=5)
            rmse_score = np.sqrt(-1 * mse_score)
            avg_rmse = np.mean(rmse_score)

            print(f'Alpha : {alpha}')
            print(f'Elastic Negative MSE score : {np.abs(np.round(mse_score, 3))}')
            print(f'Elastic RMSE scores : {np.round(rmse_score, 3)}')
            print(f'Elastic AVG RMSE : {avg_rmse:.3f}')

        else:
            print(f'Check the type values')

    # 알파 찾기 위한 gridSearchCV 수행
    def find_best_alpha(type, data, target):
        a = 0.01 # 최초 알파값
        alpha_list = [] # 알파 후보 담을 리스트

        if type == 'ridge':
            model_test = Ridge()
        elif type == 'lasso':
            model_test = Lasso()
        elif type == 'elastic':
            model_test = ElasticNet()
        else:
            print(f'Check the type values')

        for i in range(0,100):
            a = round(a, 2)
            alpha_list.append(a)
            a += 0.01

        # grid search 수행
        grid_search = GridSearchCV(model_test, param_grid={'alpha' : alpha_list})
        grid_search.fit(data, target)

        # Best alpha
        # print(f'Best alpha : {grid_search.best_params_}')

        # Best alpha & MSE & RMSE
        MSE = np.abs(grid_search.best_score_)
        RMSE = np.sqrt(np.abs(MSE))
        print(f'Type : {type} | {grid_search.best_params_} | MSE : {MSE} | RMSE: {RMSE}')


    # VIF 계산을 위한 함수
    def find_vif(data):
        vif_df = pd.DataFrame()
        vif_df["VIF Factor"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        vif_df["features"] = data.columns

        # VIF 높은 순서대로 재정렬하고 출력
        vif_df = vif_df.sort_values(by="VIF Factor", ascending=False)
        vif_df = vif_df.reset_index().drop(columns='index')
        print(vif_df)