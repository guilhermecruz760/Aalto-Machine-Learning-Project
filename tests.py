import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

def processData(file_path):
    rawData = pd.read_csv(file_path)
    print(f'Data Shape: {rawData.shape}')
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
kf = KFold(n_splits=10, shuffle=True, random_state=1)

def evaluate_model(model, X, y, kf):
    train_accuracies = []
    val_accuracies = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Custom accuracy (within 10% error)
        train_accuracy = np.mean(np.abs((y_train - y_train_pred) / y_train) < 0.10) * 100
        val_accuracy = np.mean(np.abs((y_val - y_val_pred) / y_val) < 0.10) * 100

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    return np.mean(train_accuracies), np.mean(val_accuracies)

# Initialize models
rf_model = RandomForestClassifier(random_state=1)
svm_model = SVC(kernel='linear', random_state=1)

# Evaluate both models
rf_train_acc, rf_val_acc = evaluate_model(rf_model, X, y, kf)
svm_train_acc, svm_val_acc = evaluate_model(svm_model, X, y, kf)

# Plotting the results
models = ['Random Forest', 'SVM']
train_accuracies = [rf_train_acc, svm_train_acc]
val_accuracies = [rf_val_acc, svm_val_acc]

x = np.arange(len(models))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, train_accuracies, width, label='Training Accuracy', color='blue')
bars2 = ax.bar(x + width/2, val_accuracies, width, label='Validation Accuracy', color='green')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Accuracy Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Add value annotations on the bars
for bar in bars1 + bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')

plt.show()

# Print results
print(f"Random Forest Training Accuracy: {rf_train_acc:.2f}%")
print(f"Random Forest Validation Accuracy: {rf_val_acc:.2f}%")
print(f"SVM Training Accuracy: {svm_train_acc:.2f}%")
print(f"SVM Validation Accuracy: {svm_val_acc:.2f}%")
