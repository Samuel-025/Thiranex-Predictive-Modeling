import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay

# --- PHASE 1: DATA PREPARATION ---
df = sns.load_dataset('titanic')

# Cleaning
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df.drop(columns=['deck', 'embark_town', 'alive', 'who', 'adult_male'], inplace=True, errors='ignore')

# Encoding
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['class'] = df['class'].map({'First': 1, 'Second': 2, 'Third': 3})
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- PHASE 2: MODEL TRAINING ---
print("Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- PHASE 3: EVALUATION ---
y_pred = model.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# --- PHASE 4: VISUALIZING PERFORMANCE ---

# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Perished', 'Survived'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix: Predicting Survival')
plt.tight_layout() # Ensures everything fits in the window
plt.show() # This will open the first window. Close it to see the next one.

# 2. Feature Importance
plt.figure(figsize=(10, 6))
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=importances, y=importances.index, palette='viridis')
plt.title('Which factors mattered most for survival?')
plt.xlabel('Importance Score')
plt.tight_layout() 
plt.show() # This will open the second window.