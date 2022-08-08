import tensorflow as tf
from tensorflow import keras

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("inputs/Processed_data.csv")

X = df.drop(["customerID","Churn"], axis=1)
y = df["Churn"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

print("X_train: ",X_train.shape)
print("X_test: ", X_test.shape)




model = keras.Sequential([
    keras.layers.Dense(3, input_shape=(26,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=200)


print("X_test Evaluation: ", model.evaluate(X_test, y_test))