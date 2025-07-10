#Import Libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

#Load Dataset
df = pd.read_csv("Telco_Customer_Churn.csv")

#Encoding
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])

df_encoded = pd.get_dummies(df.drop('customerID', axis=1), drop_first=True)

#Split Dataset
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Predict
y_pred = model.predict(X_test)
print(y_pred)

#Evaluation Metrics
acc = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(acc * 100))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:", cm)

cr = classification_report(y_test, y_pred)
print("Classification Report:", cr)

#Visualize Data
feature_imp = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_imp[:15], y=feature_imp.index[:15], palette='coolwarm')
plt.title("Top 15 Feature Importances - Logistic Regression")
plt.xlabel("Coefficient (Impact on Churn)")
plt.ylabel("Features")
plt.tight_layout()
plt.show()