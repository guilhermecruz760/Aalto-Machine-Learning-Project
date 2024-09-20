import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np


def processData(file_path):
    rawData = pd.read_csv(file_path)
    data = rawData.drop(['Hours_Studied', 'Attendance', 'Access_to_Resources',
                        'Previous_Scores', 'Motivation_Level', 
                        'Internet_Access', 'Tutoring_Sessions', 'Family_Income', 
                        'Teacher_Quality', 'School_Type', 'Learning_Disabilities',
                        'Parental_Education_Level', 'Distance_from_Home', 'Gender'], axis=1)
    return data

data = processData("/Users/gui/Desktop/Machine_Learning/Project/Aalto-Machine-Learning-Project/StudentPerformanceFactors.csv")
X = data[['Parental_Involvement', 'Extracurricular_Activities', 'Sleep_Hours', 'Physical_Activity', 'Peer_Influence']]
y = data['Exam_Score']


# Mapping of categorical variables to numerical values
parental_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
extracurricular_mapping = {'No': 0, 'Yes': 1}
peer_mapping = {'Positive': 0, 'Neutral': 1, 'Negative': 2}

X['Parental_Involvement'] = X['Parental_Involvement'].map(parental_mapping)
X['Extracurricular_Activities'] = X['Extracurricular_Activities'].map(extracurricular_mapping)
X['Peer_Influence'] = X['Peer_Influence'].map(peer_mapping)

# Check for and print columns with missing values
print("Missing values before imputation:")
print(X.isnull().sum())


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)


# Replace SVC with RandomForestClassifier
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

# Print classification report
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_val, y_pred)
mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

absolute_error = np.abs(y_val - y_pred)
mean_absolute_error = np.sum(absolute_error) / len(y_val)
print(f"Mean Absolute Error (MAE): {mean_absolute_error}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r_squared}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")



# 1. Define a threshold (e.g., 10% error)
threshold = 0.10

# 2. Calculate the absolute percentage error
percentage_error = np.abs((y_val - y_pred) / y_val)

# 3. Calculate accuracy: Percentage of predictions within the threshold
accuracy = np.mean(percentage_error < threshold) * 100

# 4. Print accuracy
print(f"Custom Accuracy (within {threshold*100}% error margin): {accuracy:.2f}%")

