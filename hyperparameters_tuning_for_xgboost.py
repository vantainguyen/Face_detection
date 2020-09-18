from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

data = loadtxt('pima-indians-diabetes.csv', delimiter = ',')

X = data[:,:8]
y = data[:,8]

kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 7)
estimator = XGBClassifier()
learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learning_rate = learning_rate)

grid_search = GridSearchCV(estimator,param_grid, scoring='neg_log_loss', n_jobs = -1, verbose= 1,cv = kfold)

grid_search.fit(X,y)

print('Best: %.4f at %s' %(grid_search.best_score_,grid_search.best_params_))

means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']

for mean, std, param in zip(means, stds, params):
    print('mean: %.2f, std: %.4f, param: %s' %(mean, std, param))