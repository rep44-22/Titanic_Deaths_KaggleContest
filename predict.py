import joblib 
#from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pandas as pd

df=pd.read_csv("test.csv")
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

X_test=df[['Age','Sex_male','Pclass','SibSp','Parch','Fare']]

model = joblib.load("xgb_model.pkl")

predictions=model.predict(X_test)

subs=pd.DataFrame({
    'PassengerId': df['PassengerId'],
    'Survived':predictions.astype(int)
})

subs.to_csv("submission.csv", index=False)
print("file created")