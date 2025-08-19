import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# load dataset to dataframe
credit_card_data = pd.read_csv("creditcard.csv")
credit_card_data.head()

# check number of missing values in each column
sum = credit_card_data.isnull().sum()

# distribution of legit transactions and fraudulent transactions
count = credit_card_data["Class"].value_counts()

# Comment: very unbalanced dataset -> cannot use it to feed ML model because model cannot recognise fraudulent transactions due to very less fraudulent data

# separating data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# statistical measures
ldes = legit.Amount.describe()
fdes = fraud.Amount.describe()

# compare values for both transactions
mean = credit_card_data.groupby("Class").mean()

# under-sampling: build a sample dataset containing similar distribution of normal transactions and fraudulent transactions
legit_sample = legit.sample(n=492)

new_dataset = pd.concat([legit_sample, fraud], axis=0)

n = new_dataset["Class"].value_counts()

new_mean = new_dataset.groupby(
    "Class"
).mean()  # good sample occurs when the mean before and after sample isnt too different

# splitting the data into features and types
X = new_dataset.drop(columns="Class", axis=1)
Y = new_dataset["Class"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, Y_train)

# Model evaluation
X_train_prediction = model.predict(X_scaled)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print(f'training_data_accuracy: {training_data_accuracy}')

X_test_scaled = scaler.transform(X_test)
X_test_prediction = model.predict(X_test_scaled)

print(f'test_data_accuracy: {accuracy_score(Y_test, X_test_prediction)}')
print(f"Test precision: {precision_score(Y_test, X_test_prediction):.4f}")
print(f"Test recall: {recall_score(Y_test, X_test_prediction):.4f}")
print(f"Test F1 score: {f1_score(Y_test, X_test_prediction):.4f}")
