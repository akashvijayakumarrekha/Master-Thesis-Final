import time
import os
import psutil
import dill as pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, f1_score, accuracy_score, recall_score
from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
from concrete.ml.sklearn import RandomForestClassifier as ConcreteRandomForestClassifier
from concrete.ml.sklearn import XGBClassifier as ConcreteXGBClassifier
from concrete.ml.sklearn import LinearSVC as ConcreteLinearSVC
from pathlib import Path
from sklearn.utils import shuffle

def save_model(model, filename):
    with Path(filename).open("w") as f:
        model.dump(f)
    return os.path.getsize(filename) / 1024 / 1024  # Size in megabytes

def load_model(filename):
    from concrete.ml.common.serialization.loaders import load
    with Path(filename).open("r") as f:
        model = load(f)
    return model

def save_object(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)
    return os.path.getsize(filename) / 1024 / 1024  # Size in megabytes

def load_object(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
    

def evaluate_model(name, y_test, predictions, training_time):
    accuracy = accuracy_score(y_test, predictions)
    precision = average_precision_score(y_test, predictions, average='macro', zero_division=0)
    recall = recall_score(y_test, predictions, average='macro', zero_division=0)
    f1 = f1_score(y_test, predictions, average='macro', zero_division=0)
    error_rate = 1 - accuracy

    print(f"{name}:")
    print(f"Training Time: {training_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Error Rate: {error_rate:.4f}\n")

def evaluate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = average_precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    error_rate = 1 - accuracy
    return accuracy, precision, f1, error_rate

def print_metrics(size, metrics, memory_usage, duration, throughput):
    print(f'Model Size: {size} MB')
    print(f'Accuracy: {metrics[0]}, Precision: {metrics[1]}, F1 Score: {metrics[2]}, Error Rate: {metrics[3]}')
    print(f'Testing Memory Usage: {memory_usage} MB')
    print(f'Latency: {duration:.2f} ms per sample')
    print(f'Throughput: {throughput:.2f} samples per second')

# Initial setup

process = psutil.Process(os.getpid())

# Load data
dataset_csv_path = '/home/akash/Desktop/MasterThesis/NF-CSE-CIC-IDS2018.csv'
data = pd.read_csv(dataset_csv_path)
no_of_samples = 1000
data_benign = data[data['label'] == 0].sample(int(no_of_samples/2), random_state=42)
data_attack = data[data['label'] == 1].sample(int(no_of_samples/2), random_state=42)

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

# Initialize the scikit models
sci_models = {
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=9, learning_rate=0.2, subsample=1, 
        colsample_bytree=0.75, gamma=0, min_child_weight=1, reg_alpha=0.1, 
        reg_lambda=2),    
    "Random Forest": RandomForestClassifier(
        n_estimators=20, criterion='gini', max_depth=4, 
        min_samples_split=2, min_samples_leaf=1, max_features='sqrt', 
        random_state=42),        
    "Logistic Regression": LogisticRegression(
        penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
        intercept_scaling=1, class_weight=None, random_state=None, 
        solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, 
        warm_start=False, n_jobs=None, l1_ratio=None),
    "LinearSVC": LinearSVC(
        penalty='l2', loss='squared_hinge', dual=True, tol=0.001, C=0.5, 
        multi_class='ovr', fit_intercept=True, intercept_scaling=1, 
        class_weight='balanced', verbose=0, random_state=None, max_iter=2000)
}

# Train and predict with each scikit model
for name, model in sci_models.items():
    start_time = time.time()  
    model.fit(x_train, y_train)
    end_time = time.time() 

    training_time = end_time - start_time  

    predictions = model.predict(x_test)

    
    evaluate_model(name, y_test, predictions, training_time)


# Initialize the concrete ml models
concrete_models = [
    (ConcreteXGBClassifier, {
        'n_estimators': 200,
        'max_depth': 9,
        'learning_rate': 0.2,
        'subsample': 1,
        'colsample_bytree': 0.75,
        'gamma': 0,
        'min_child_weight': 1,
        'reg_alpha': 0.1,
        'reg_lambda': 2
    }),
    (ConcreteRandomForestClassifier, {
        'n_bits': 6,
        'n_estimators': 20,
        'criterion': 'gini',
        'max_depth': 4,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': 42
    }),    
    (ConcreteLogisticRegression, {
        'n_bits': 8,
        'penalty': 'l2',
        'dual': False,
        'tol': 0.0001,
        'C': 1.0,
        'fit_intercept': True,
        'intercept_scaling': 1,
        'class_weight': None,
        'random_state': None,
        'solver': 'lbfgs',
        'max_iter': 100,
        'multi_class': 'auto',
        'verbose': 0,
        'warm_start': False,
        'n_jobs': None,
        'l1_ratio': None
    }),
    (ConcreteLinearSVC, {
        'n_bits': 8,
        'penalty': 'l2',
        'loss': 'squared_hinge',
        'dual': True,
        'tol': 0.001,
        'C': 0.5,
        'multi_class': 'ovr',
        'fit_intercept': True,
        'intercept_scaling': 1,
        'class_weight': 'balanced',
        'verbose': 0,
        'random_state': None,
        'max_iter': 2000
    })
]


for model_cls, kwargs in concrete_models:
    # Unencrypted Model training
    print(f"\nTraining {model_cls.__name__}...")
    model = model_cls(**kwargs)
    start_time = time.time()
    model, sklearn_model = model.fit_benchmark(x_train, y_train)
    print(f"Training time: {time.time() - start_time:.2f} seconds")
    
    memory_before = process.memory_info().rss / 1024 / 1024
    start_time = time.time()
    y_pred = sklearn_model.predict(x_test)
    metrics = evaluate_metrics(y_test, y_pred)
    memory_after = process.memory_info().rss / 1024 / 1024
    print_metrics(save_object(model, f'unencrypted_model_{model_cls.__name__}.pkl'), metrics, memory_after - memory_before, 
                  (time.time() - start_time) * 1000 / len(x_test), len(x_test) / (time.time() - start_time))

    # FHE model preparation
    print("Compiling...")
    compilation_time_start = time.time()
    memory_before_compilation = process.memory_info().rss / 1024 / 1024  # Convert bytes to megabytes
    circuit = model.compile(x_train)
    compilation_duration = time.time() - compilation_time_start
    memory_after_compilation = process.memory_info().rss / 1024 / 1024  # Convert bytes to megabytes
    memory_used = memory_after_compilation - memory_before_compilation
    print(f"Compilation time: {compilation_duration:.2f} seconds")
    
    print("Generating keys...")
    keygen_time_start = time.time()
    circuit.client.keygen(force=False)
    keygen_duration = time.time() - keygen_time_start
    print(f"Key generation time: {keygen_duration:.2f} seconds")
    
    # FHE predictions
    memory_before = process.memory_info().rss / 1024 / 1024
    x_test_sample = x_test
    y_test_sample = y_test
    predict_time_start = time.time()
    y_pred_fhe = model.predict(x_test_sample, fhe="execute")
    predict_duration = time.time() - predict_time_start
    metrics = evaluate_metrics(y_test_sample, y_pred_fhe)
    memory_after = process.memory_info().rss / 1024 / 1024
    print_metrics(memory_used, metrics, memory_after - memory_before, 
                  predict_duration * 1000 / len(x_test_sample), len(x_test_sample) / predict_duration)

