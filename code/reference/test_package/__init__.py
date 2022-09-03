import sched
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, GridSearchCV

# Ridge, Lasso, ElasticNet의 MSE, RMSE 스코어 계산하는 함수
def score_checker(type, data, target, alpha):
    if type == 'ridge':
        type_x = Ridge(alpha)
    elif type == 'lasso':
        type_x = Lasso(alpha)
    elif type == 'elastic':
        type_x = ElasticNet(alpha)
    else:
        print(f'Check the type values')

    mse_score = cross_val_score(type_x, data, target, scoring="neg_mean_squared_error", cv=5)
    rmse_score = np.sqrt(-1 * mse_score)
    avg_rmse = np.mean(rmse_score)

    print(f'Alpha : {alpha}')
    print(f'{type} Negative MSE score : {np.abs(np.round(mse_score, 3))}')
    print(f'{type} RMSE scores : {np.round(rmse_score, 3)}')
    print(f'{type} AVG RMSE : {avg_rmse:.3f}\n')

# 노가다를 줄여봅시다
def score_many_checker(data, target, alpha_ridge, alpha_rasso, alpha_elastic):
    type_list = ['ridge', 'lasso', 'elastic']
    alpha = [alpha_ridge, alpha_rasso, alpha_elastic]
    for i in range(0, len(type_list)):
        score_checker(type_list[i], data, target, alpha[i])

# 향후 개선안
# 밑에 있는 find_best_alpha랑 연계해서 score_many_checker 돌리는 게 가능할까...?

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


# coeff 확인
def check_coeff(type, data, target, alpha):
    flg, axs = plt.subplots(figsize=(18,6), nrows=1, ncols=5)
    coeff_df = pd.DataFrame() #alpha 회귀 계수 저장용

    alphas = alpha

    for pos, alpha in enumerate(alphas):
        if type == 'ridge':
            type_x = Ridge(alpha = alpha)
        elif type == 'lasso':
            type_x = Lasso(alpha = alpha)
        elif type == 'elastic':
            type_x = ElasticNet(alpha = alpha)
        else:
            print(f'Check the type values')
            break

        type_x.fit(data, target)

        # alpha에 따른 피처별 회귀 변수를 Series로 받고 데이터프레임에 추가
        coeff = pd.Series(data = type_x.coef_, index=data.columns)
        colname = 'alpha: ' + str(alpha)
        coeff_df[colname] = coeff

        # 막대그래프에서 회귀계수 높은 순서대로 시각화
        coeff = coeff.sort_values(ascending=False)
        axs[pos].set_title(colname)
        axs[pos].set_xlim(-3,6)
        sns.barplot(x=coeff.values, y=coeff.index, ax=axs[pos])

    plt.show()

    type_x_alpha = alphas
    sort_column = 'alpha: '+str(type_x_alpha[0])
    print(coeff_df.sort_values(by=sort_column, ascending=False))


def alpha_master(type, data, target):
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

    # Get Best Alpha
    best_alpha = grid_search.best_params_['alpha']

    # Is it right?
    score_checker(type, data, target, best_alpha)