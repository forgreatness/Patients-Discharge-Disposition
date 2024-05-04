import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as srn
import tensorflow as tf

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import normalize, StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE, RandomOverSampler
from ast import literal_eval  # To convert string to list
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from scikeras.wrappers import KerasRegressor


smote = SMOTE()
randomSampler = RandomOverSampler()
scaler = StandardScaler()


def augmentData(encounters):
    # Number of synthetic samples to generate
    num_synthetic_samples = 3

    # Generate synthetic samples
    synthetic_samples = []
    for i in range(num_synthetic_samples):
        # Perturb original data slightly
        synthetic_sample = encounters.apply(lambda x: x + np.random.normal(scale=0.1), axis=0)
        synthetic_samples.append(synthetic_sample)

    # Concatenate original and synthetic samples
    augmented_data = pd.concat([encounters] + synthetic_samples, ignore_index=True)
    
    return augmented_data


encounters = pd.read_csv('./finalEncounterData.csv', index_col=0)

y = encounters['Duration of Care']  # Assuming 'Duration of Care' is the target variable
x = encounters.drop('Duration of Care', axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

"""      Feature Scaling       """
# X_train = normalize(X_train, axis=0)
# X_test = normalize(X_test, axis=0)

# Defined Model Architecture
model = Sequential([
    Dense(24, activation='relu', kernel_regularizer=regularizers.l1(0.01), input_shape=(X_train.shape[1],)),
    Dense(20, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    Dense(1)
])

# Compile Neural Network Model
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1)
loss = model.evaluate(X_test, y_test)

# Predict future encounter cases
future_encounter_predictions = model.predict(X_test)
rounded_predictions = np.ceil(future_encounter_predictions).astype(int)

print("Predictions for future encounter cases:", rounded_predictions)
print("Actual value for future encounter cases:\n", y_test)
# Calculate regression metrics
mse = mean_squared_error(y_test, future_encounter_predictions)
mae = mean_absolute_error(y_test, future_encounter_predictions)
r2 = r2_score(y_test, future_encounter_predictions)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)








"""               Steps to Complete
1. Hyper parameter tuning
2. Feature scaling 
    a. When should feature scaling be specifically use
    b. Does it apply to our cases
    c. Which feature scaling is the one to use
3. Are there any good data augmentation technique that would benefit us
4. Can we try batch norm
5. What is RNN and CNN and how could they help
6. What are somethings we can do to improve our model


"""
# Define the parameter grid
# Function to create Keras model
# Define the Keras model function
# build function has issue here
# def build_clf(unit): 
#   # creating the layers of the NN 
#   ann = tf.keras.models.Sequential([
#     Dense(24, activation='relu', kernel_regularizer=regularizers.l1(0.01), input_shape=(X_train.shape[1],)),
#     Dense(20, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
#     Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
#     Dense(1)
#   ])
#   return ann

# model=KerasRegressor(build_fn=build_clf)
# print(model.get_params().keys())

# # Define the parameter grid
# params={'batch_size':[100, 20, 50, 25, 32],  
#         'epochs':[200, 100, 300, 400], 
#         'validation_split': [0.1, 0.2]           
#         } 
# gs=GridSearchCV(estimator=model, param_grid=params, cv=10) 
# # now fit the dataset to the GridSearchCV object.  
# gs = gs.fit(X_train, y_train)


# best_params=gs.best_params_ 
# accuracy=gs.best_score_ 







"""      Trying Random Forest       """
randomForestModel = RandomForestRegressor(n_estimators=100, random_state=42)
randomForestModel.fit(X_train, y_train)
# Make predictions on the test set
rfYPred = randomForestModel.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, rfYPred)
mae = mean_absolute_error(y_test, rfYPred)
r2 = r2_score(y_test, rfYPred)

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)