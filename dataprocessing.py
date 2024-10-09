# Import Necessary Libraries
import pandas as pd
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_absolute_percentage_error)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Function to Process Data
def processData(file_path):
    """
    Loads and processes the dataset by dropping irrelevant features.
    
    Parameters:
    - file_path (str): Path to the CSV file containing the dataset.
    
    Returns:
    - pd.DataFrame: Processed dataset.
    """
    rawData = pd.read_csv(file_path)
    print(f'Data Shape: {rawData.shape}')
    # Dropping columns not used in the analysis
    data = rawData.drop(['Hours_Studied', 'Attendance', 'Access_to_Resources',
                         'Previous_Scores', 'Motivation_Level', 
                         'Internet_Access', 'Tutoring_Sessions', 'Family_Income', 
                         'Teacher_Quality', 'School_Type', 'Learning_Disabilities',
                         'Parental_Education_Level', 'Distance_from_Home', 'Gender'], axis=1)
    print(f'Data Shape after dropping columns: {data.shape}')
    return data

# Path to the dataset
file_path = "/Users/gui/Desktop/Machine_Learning/Project/Aalto-Machine-Learning-Project/StudentPerformanceFactors.csv"

# Load and Process Data
data = processData(file_path)

# Feature Selection
X = data[['Parental_Involvement', 'Extracurricular_Activities', 'Sleep_Hours', 
          'Physical_Activity', 'Peer_Influence']]
y = data['Exam_Score']

# Encode Categorical Variables
categorical_features = ['Parental_Involvement', 'Extracurricular_Activities', 'Peer_Influence']
label_encoders = {}
for column in categorical_features:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le
    print(f'Encoded {column}: {X[column].unique()}')

# Split Data into Training and Testing Sets (80% Train, 20% Test)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f'Training Set Shape: {X_train_full.shape}')
print(f'Testing Set Shape: {X_test.shape}')

# Implement K-Fold Cross-Validation on the Training Set
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# Initialize Dictionaries to Store Metrics for Each Model
metrics = {
    'SVR': {'train_mae': [], 'val_mae': [], 'train_mse': [], 'val_mse': [],
            'train_rmse': [], 'val_rmse': [], 'train_r2': [], 'val_r2': [],
            'train_mape': [], 'val_mape': [], 'train_accuracy': [], 'val_accuracy': []},
    'DecisionTree': {'train_mae': [], 'val_mae': [], 'train_mse': [], 'val_mse': [],
                    'train_rmse': [], 'val_rmse': [], 'train_r2': [], 'val_r2': [],
                    'train_mape': [], 'val_mape': [], 'train_accuracy': [], 'val_accuracy': []},
    'RandomForest': {'train_mae': [], 'val_mae': [], 'train_mse': [], 'val_mse': [],
                    'train_rmse': [], 'val_rmse': [], 'train_r2': [], 'val_r2': [],
                    'train_mape': [], 'val_mape': [], 'train_accuracy': [], 'val_accuracy': []}
}

# Define Models with Hyperparameters
models = {
    'SVR': SVR(kernel='linear', C=1.0, epsilon=0.1),
    'DecisionTree': DecisionTreeRegressor(random_state=1, max_depth=None),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=-1)
}

# Function to Calculate Custom Accuracy (Within 10% Error Margin)
def calculate_custom_accuracy(y_true, y_pred, threshold=0.1):
    """
    Calculates the percentage of predictions within a specified error margin.
    
    Parameters:
    - y_true (np.array): True target values.
    - y_pred (np.array): Predicted target values.
    - threshold (float): Error margin threshold (default is 0.1 for 10%).
    
    Returns:
    - float: Percentage of predictions within the error margin.
    """
    percentage_error = np.abs((y_true - y_pred) / y_true)
    accuracy = np.mean(percentage_error < threshold) * 100
    return accuracy

# Perform K-Fold Cross-Validation
fold = 1
for train_index, val_index in kf.split(X_train_full):
    print(f'\nProcessing Fold {fold}...')
    X_train, X_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
    y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]
    
    for model_name, model in models.items():
        # Train the Model
        model.fit(X_train, y_train)
        
        # Make Predictions on Training and Validation Sets
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate Metrics for Training Set
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
        train_accuracy = calculate_custom_accuracy(y_train, y_train_pred)
        
        # Calculate Metrics for Validation Set
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = np.sqrt(val_mse)
        val_r2 = r2_score(y_val, y_val_pred)
        val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
        val_accuracy = calculate_custom_accuracy(y_val, y_val_pred)
        
        # Append Metrics to the Dictionaries
        metrics[model_name]['train_mae'].append(train_mae)
        metrics[model_name]['val_mae'].append(val_mae)
        metrics[model_name]['train_mse'].append(train_mse)
        metrics[model_name]['val_mse'].append(val_mse)
        metrics[model_name]['train_rmse'].append(train_rmse)
        metrics[model_name]['val_rmse'].append(val_rmse)
        metrics[model_name]['train_r2'].append(train_r2)
        metrics[model_name]['val_r2'].append(val_r2)
        metrics[model_name]['train_mape'].append(train_mape)
        metrics[model_name]['val_mape'].append(val_mape)
        metrics[model_name]['train_accuracy'].append(train_accuracy)
        metrics[model_name]['val_accuracy'].append(val_accuracy)
    
    fold += 1

# Calculate Average Metrics Across All Folds
average_metrics = {}
for model_name in models.keys():
    average_metrics[model_name] = {metric: np.mean(scores) for metric, scores in metrics[model_name].items()}

# Display the Average Metrics
print("\n\n=== Average Metrics Across All Folds ===")
for model_name, metrics_dict in average_metrics.items():
    print(f"\nModel: {model_name}")
    print(f"Training MAE: {metrics_dict['train_mae']:.4f}")
    print(f"Validation MAE: {metrics_dict['val_mae']:.4f}")
    print(f"Training MSE: {metrics_dict['train_mse']:.4f}")
    print(f"Validation MSE: {metrics_dict['val_mse']:.4f}")
    print(f"Training RMSE: {metrics_dict['train_rmse']:.4f}")
    print(f"Validation RMSE: {metrics_dict['val_rmse']:.4f}")
    print(f"Training R²: {metrics_dict['train_r2']:.4f}")
    print(f"Validation R²: {metrics_dict['val_r2']:.4f}")
    print(f"Training MAPE: {metrics_dict['train_mape']:.2f}%")
    print(f"Validation MAPE: {metrics_dict['val_mape']:.2f}%")
    print(f"Training Custom Accuracy (±10%): {metrics_dict['train_accuracy']:.2f}%")
    print(f"Validation Custom Accuracy (±10%): {metrics_dict['val_accuracy']:.2f}%")

# Determine the Best Model Based on Validation MAPE (Lower is Better)
best_model_name = min(average_metrics, key=lambda x: average_metrics[x]['val_mape'])
print(f"\n\nBest Model Selected: {best_model_name}")

# Initialize the Final Model with Best Parameters (Reinitialize to avoid data leakage)
if best_model_name == 'SVR':
    final_model = SVR(kernel='linear', C=1.0, epsilon=0.1)
elif best_model_name == 'DecisionTree':
    final_model = DecisionTreeRegressor(random_state=1, max_depth=None)
elif best_model_name == 'RandomForest':
    final_model = RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=-1)
else:
    raise ValueError("Unknown Model Selected")

# Train the Final Model on the Entire Training Set
final_model.fit(X_train_full, y_train_full)

# Make Predictions on the Test Set
y_test_pred = final_model.predict(X_test)

# Calculate Test Metrics
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)
test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
test_accuracy = calculate_custom_accuracy(y_test, y_test_pred)

# Display Test Metrics
print(f"\n=== Final Model Evaluation on Test Set ===")
print(f"Model: {best_model_name}")
print(f"Test Mean Absolute Error (MAE): {test_mae:.4f}")
print(f"Test Mean Squared Error (MSE): {test_mse:.4f}")
print(f"Test Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"Test R-squared (R²): {test_r2:.4f}")
print(f"Test Mean Absolute Percentage Error (MAPE): {test_mape:.2f}%")
print(f"Test Custom Accuracy (±10%): {test_accuracy:.2f}%")

# Visualize Validation Custom Accuracy Across Folds for Each Model
plt.figure(figsize=(12, 8))
for model_name in models.keys():
    plt.plot(range(1, 11), metrics[model_name]['val_accuracy'], marker='o', label=model_name)
plt.title('Validation Custom Accuracy (±10% Error) per Fold for Different Models')
plt.xlabel('Fold')
plt.ylabel('Validation Custom Accuracy (%)')
plt.legend()
plt.grid(True)
plt.xticks(range(1, 11))
plt.show()

# Visualize Comparison of Average MAPE for Each Model
average_mape = {model: metrics[model]['val_mape'] for model in models.keys()}
models_list = list(average_mape.keys())
mape_values = list(average_mape.values())

plt.figure(figsize=(8, 6))
bars = plt.bar(models_list, mape_values, color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Comparison of Average Validation MAPE Across Models')
plt.xlabel('Model')
plt.ylabel('Average Validation MAPE (%)')
plt.ylim(0, max(mape_values) + 10)

# Annotate Bars with MAPE Values
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}%', 
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')

plt.show()
