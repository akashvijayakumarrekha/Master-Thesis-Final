
from concrete.ml.sklearn import RandomForestClassifier, XGBClassifier, LogisticRegression, SVC
import pandas as pd
from sklearn.utils import shuffle
import itertools
from sklearn.model_selection import train_test_split



# Load data
dataset_csv_path = '/home/akash/Desktop/MasterThesis/NF-CSE-CIC-IDS2018.csv'
data = pd.read_csv(dataset_csv_path)
data_benign = data[data['label'] == 0].sample(n=50000, random_state=42)
data_attack = data[data['label'] == 1].sample(n=50000, random_state=42)

# Split each class into training and testing
x_train_benign, x_test_benign, y_train_benign, y_test_benign = train_test_split(
    data_benign.drop(columns=['label']), data_benign['label'], test_size=0.15, random_state=42)

x_train_attack, x_test_attack, y_train_attack, y_test_attack = train_test_split(
    data_attack.drop(columns=['label']), data_attack['label'], test_size=0.15, random_state=42)

# Combine the splits from each class
x_train = pd.concat([x_train_benign, x_train_attack])
y_train = pd.concat([y_train_benign, y_train_attack])
x_test = pd.concat([x_test_benign, x_test_attack])
y_test = pd.concat([y_test_benign, y_test_attack])

# Optional: Shuffle the training and test data to ensure random mixing
x_train, y_train = shuffle(x_train, y_train, random_state=42)
x_test, y_test = shuffle(x_test, y_test, random_state=42)


param_grid_rf = {
    'n_bits': [4, 5, 6, 7, 8],
    'n_estimators': [10, 20, 50, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 4, 6, 8, 10],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'random_state': [42]
}

param_grid_xgb = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [3, 6, 9, 12],
    'learning_rate': [0.05, 0.1, 0.2, 0.3],
    'subsample': [0.5, 0.75, 1],
    'colsample_bytree': [0.5, 0.75, 1],
    'gamma': [0, 0.1, 0.2, 0.4],
    'min_child_weight': [1, 2, 4],
    'reg_alpha': [0, 0.1, 0.2],
    'reg_lambda': [1, 2, 4]
}

param_grid_lr = {
    'n_bits': [6, 7, 8, 9, 10],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'dual': [False],
    'tol': [0.00001, 0.0001, 0.001],
    'C': [0.5, 1.0, 2.0, 5.0],
    'fit_intercept': [True],
    'intercept_scaling': [0.5, 1, 2],
    'class_weight': [None, 'balanced'],
    'random_state': [None, 42],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [50, 100, 200, 400],
    'multi_class': ['auto', 'ovr', 'multinomial'],
    'verbose': [0, 1],
    'warm_start': [False],
    'n_jobs': [None, 1, -1],
    'l1_ratio': [None, 0.5, 1]
}

param_grid_svc = {
    'n_bits': [6, 7, 8, 9, 10],
    'penalty': ['l2'],
    'loss': ['hinge', 'squared_hinge'],
    'dual': [True, False],
    'tol': [0.0001, 0.001, 0.01],
    'C': [0.1, 0.5, 1.0, 2.0],
    'multi_class': ['ovr','crammer_singer'],
    'fit_intercept': [True],
    'intercept_scaling': [0.5, 1, 2],
    'class_weight': [None, 'balanced'],
    'verbose': [0,1],
    'random_state': [None,42],
    'max_iter': [1000, 2000, 3000]
}

# Function for Grid search
def perform_grid_search(model_class, param_grid, x_train, y_train):
    best_score = float('-inf')
    best_params = {}
    for params in itertools.product(*param_grid.values()):
        model = model_class(**dict(zip(param_grid.keys(), params)))
        model.fit(x_train, y_train)
        score = model.score(x_train, y_train)  
        if score > best_score:
            best_score = score
            best_params = dict(zip(param_grid.keys(), params))
    return best_params

# Perform grid search for each model
best_params_rf = perform_grid_search(RandomForestClassifier, param_grid_rf, x_train, y_train)
print("Best parameters for RF:", best_params_rf)

best_params_xgb = perform_grid_search(XGBClassifier, param_grid_xgb, x_train, y_train)
print("Best parameters for XGB:", best_params_xgb)

best_params_lr = perform_grid_search(LogisticRegression, param_grid_lr, x_train, y_train)
print("Best parameters for LR:", best_params_lr)

best_params_svc = perform_grid_search(SVC, param_grid_svc, x_train, y_train)
print("Best parameters for SVC:", best_params_svc)
