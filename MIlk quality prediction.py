import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Read the dataset
file_path = r"C:\Users\User\Desktop\constantinos\sports data analytics\2.Sports Analytics School - Programming Full Course Training 2023\9.Practical Examples on data analysis\datasets\Milk Quality Prediction\milknew.csv"
df = pd.read_csv(file_path)

# Step 3: Explore the dataset
print("Dataset Information:")
print(df.info())
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Step 4: Data Preprocessing
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Remove unnecessary transformations for binary columns
X = df.drop(['Grade'], axis=1)
y = df['Grade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Build a Predictive Model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

# Step 6: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_rep)

# Step 7: Save the Model (Optional)
model_save_path = r"C:\Users\User\Desktop\constantinos\sports data analytics\2.Sports Analytics School - Programming Full Course Training 2023\9.Practical Examples on data analysis\milk_quality_model.joblib"
joblib.dump(rf_classifier, model_save_path)

# Step 8: Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=rf_classifier.classes_, yticklabels=rf_classifier.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Step 9: Summary and Documentation
"""
Script for predicting milk quality using machine learning.

Steps:
1. Import necessary libraries
2. Read the dataset
3. Explore the dataset
4. Data preprocessing
5. Build a predictive model (Random Forest Classifier)
6. Evaluate the model
7. Save the model (optional)
8. Confusion Matrix Plot
9. Summary and Documentation
"""
