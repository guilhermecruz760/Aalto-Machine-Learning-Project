import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np


def processData(file_path):
    rawData = pd.read_csv(file_path)
    print(f'Data Shape:{rawData.shape}')
    data = rawData.drop(['Hours_Studied', 'Attendance', 'Access_to_Resources',
                        'Previous_Scores', 'Motivation_Level', 
                        'Internet_Access', 'Tutoring_Sessions', 'Family_Income', 
                        'Teacher_Quality', 'School_Type', 'Learning_Disabilities',
                        'Parental_Education_Level', 'Distance_from_Home', 'Gender'], axis=1)
    
    return data

data = processData("/Users/gui/Desktop/Machine_Learning/Project/Aalto-Machine-Learning-Project/StudentPerformanceFactors.csv")
X = data[['Parental_Involvement', 'Extracurricular_Activities', 'Sleep_Hours', 'Physical_Activity', 'Peer_Influence']]
y = data['Exam_Score']

label_encoders = {}
for column in ['Parental_Involvement', 'Extracurricular_Activities', 'Peer_Influence']:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Check for missing values
print("Missing values before imputation:")
print(X.isnull().sum())

# Implement K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

mae_scores = []
mse_scores = []
rmse_scores = []
r_squared_scores = []
mape_scores = []
custom_accuracy_scores = []

# Loop through each fold
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Train the model
    model = RandomForestClassifier(random_state=1)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_val)

    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_val, y_pred)
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
    absolute_error = np.abs(y_val - y_pred)
    mean_absolute_error = np.sum(absolute_error) / len(y_val)

    # Custom accuracy (within 10% error)
    threshold = 0.10
    percentage_error = np.abs((y_val - y_pred) / y_val)
    accuracy = np.mean(percentage_error < threshold) * 100

    # Append scores
    mae_scores.append(mean_absolute_error)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    r_squared_scores.append(r_squared)
    mape_scores.append(mape)
    custom_accuracy_scores.append(accuracy)

# Print average metrics across all folds
print(f"Mean Absolute Error (MAE): {np.mean(mae_scores)}")
print(f"Mean Squared Error (MSE): {np.mean(mse_scores)}")
print(f"Root Mean Squared Error (RMSE): {np.mean(rmse_scores)}")
print(f"R-squared (R²): {np.mean(r_squared_scores)}")
print(f"Mean Absolute Percentage Error (MAPE): {np.mean(mape_scores)}%")
print(f"Custom Accuracy (within 10% error margin): {np.mean(custom_accuracy_scores):.2f}%")
