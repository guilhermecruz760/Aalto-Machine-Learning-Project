import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


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


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Replace SVC with RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

# Print classification report
print(classification_report(y_val, y_pred, zero_division=0))
print(f"Accuracy: {accuracy_score(y_val, y_pred)}")
