import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

#load data
data = pd.read_csv('/kaggle/input/datasets/brendan45774/test-file/tested.csv')

#clean & map data
data.drop('Cabin',axis =1)
data['Age'] = data['Age'].fillna(data["Age"].mean())
data["Sex"] = data['Sex'].map({'male': 0, 'female': 1})

#feature Selection
x = data[['Age','Sex','SibSp','Parch','Pclass']]
y = data['Survived']

#Split data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

#Train model
model= LogisticRegression()
model.fit(x_train,y_train)

#predict outcome
predictions = model.predict(x_test)

#Accuracy
accuracy = accuracy_score(y_test,predictions)
print(accuracy)

joblib.dump(model,'titanic_model1.pkl')