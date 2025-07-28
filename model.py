from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

df=pd.read_csv("train.csv")
print(df.head())
enc=OneHotEncoder()

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

X=df[['Age','Sex_male','Pclass','SibSp','Parch','Fare']]
y=df['Survived']

X_train,X_test, y_train,y_test=train_test_split(X,y, test_size=0.3,random_state=42)

model=XGBClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

import joblib
joblib.dump(model, "xgb_model.pkl")