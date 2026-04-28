import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("dataset.csv")

data = data.dropna()

le_has_car = LabelEncoder()
le_Department = LabelEncoder()
le_work_mode = LabelEncoder()
le_will_go = LabelEncoder()

data["Has_car"] = le_has_car.fit_transform(data["Has_car"])
data["Department"] = le_Department.fit_transform(data["Department"])
data["Work_mode"] = le_work_mode.fit_transform(data["Work_mode"])
data["Will_go"] = le_will_go.fit_transform(data["Will_go"])

X = data.drop("Will_go", axis=1)
y = data["Will_go"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: ", accuracy)


punetor_ri = pd.DataFrame({
    "Distance_km": [10],
    "Has_car": le_has_car.transform(["Yes"]),
    "Department": le_Department.transform(["IT"]),
    "Work_mode": le_work_mode.transform(["Remote"]),
    "Years_At_Work": [2]
})

result = model.predict(punetor_ri)


print(data.head())
print("Prediction: ", le_will_go.inverse_transform(result)[0])





