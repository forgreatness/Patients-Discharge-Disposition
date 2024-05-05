import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as srn
import tensorflow as tf

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import normalize, StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE, RandomOverSampler
from ast import literal_eval  # To convert string to list
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam


smote = SMOTE()
randomSampler = RandomOverSampler()
scaler = StandardScaler()

encounters = pd.read_csv('./finalEncounterData.csv', index_col=0)

y = encounters['Duration of Care']  # Assuming 'Duration of Care' is the target variable
x = encounters.drop('Duration of Care', axis=1)

"""      One Hot Encoding         """
# Columns to encode
def oneHotEncodeNominalData():
    columns_to_encode = ['Service Type', 'Gender', 'Ethnicity']

    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(x[columns_to_encode])
    #Create a DataFrame with the one-hot encoded columns
    #We use get_feature_names_out() to get the column names for the encoded data
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(columns_to_encode))
    one_hot_df = one_hot_df.astype(int)

    # Concatenate the one-hot encoded dataframe with the original dataframe
    df_encoded = pd.concat([x, one_hot_df], axis=1)
    # Drop the original categorical columns
    df_encoded = df_encoded.drop(columns_to_encode, axis=1)
    
    return df_encoded

# x = oneHotEncodeNominalData()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

"""      Feature Scaling       """
# Fit the scaler to your data and transform it
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# X_train = normalize(X_train, axis=0)
# X_test = normalize(X_test, axis=0)

"""  
    Regularization | Batchnorm | Dropout
Regularization: Penalizing the loss function by making weights for some feature go to zero. Good for introducing sparsity in model ()
Dropout: The purpose is to help capture more robust correlation between data, making the model not rely too much on certian neuron for prediction.
Batch Norm: normalize the input to each layer, by calculating standard deviation. The purpose helps accerlate learning
"""

# Defined Model Architecture
model = Sequential([
    Dense(50, activation='relu', kernel_regularizer=regularizers.l1(0.01), input_shape=(X_train.shape[1],)),
    # BatchNormalization(),
    # Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    # BatchNormalization(),
    # Dropout(0.7),
    Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    Dense(1)
])

# Compile Neural Network Model
learning_rate = 0.1  # Choose your desired learning rate
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer='Adam', loss='mean_squared_error')
model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1)
loss = model.evaluate(X_test, y_test)

# Predict future encounter cases
future_encounter_predictions = model.predict(X_test)
rounded_predictions = np.round(future_encounter_predictions).astype(int)

# Assuming rounded_predictions and y_test are pandas Series
rounded_predictions = rounded_predictions.flatten()  # Flatten the array if it's not already one-dimensional
# Now create the DataFrame
combined_df = pd.DataFrame({'Predictions': rounded_predictions, 'Actual': y_test})
print("predictions and actual tested encounters side by side\n", combined_df)

# Calculate regression metrics
mse = mean_squared_error(y_test, future_encounter_predictions)
mae = mean_absolute_error(y_test, future_encounter_predictions)
r2 = r2_score(y_test, future_encounter_predictions)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)








"""               Steps to Complete
1. Hyper parameter tuning ✅
2. Feature scaling ✅
    a. When should feature scaling be specifically use
    b. Does it apply to our cases
    c. Which feature scaling is the one to use
3. Are there any good data augmentation technique that would benefit us
4. Can we try batch norm ✅
5. What is RNN and CNN and how could they help
6. What are somethings we can do to improve our model
"""



######         Hyper Parameter Tuning and cross validation with GridSearchCV                      

"""     scikeras.wrappers import KerasRegressor
⭐ Use this article to migrate the parameter tuning to most recent
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
"""
# Define the parameter grid
# Function to create Keras model
# Define the Keras model function

# from tensorflow.keras.metrics import R2Score
# from scikeras.wrappers import KerasRegressor

# def get_reg(meta, hidden_layer_sizes, dropout):
#     model = Sequential()
#     model.add(Input(shape=(X_train.shape[1],)))
#     for hidden_layer_size in hidden_layer_sizes:
#         model.add(Dense(hidden_layer_size, activation="relu"))
#         model.add(Dropout(dropout))
#     model.add(Dense(1))
#     return model

# reg = KerasRegressor(
#     model=get_reg,
#     loss="mse",
#     metrics=[R2Score],
#     hidden_layer_sizes=(100,),
#     dropout=0.5,
# )

# reg.fit(X_train, y_train)
# print(reg.get_params())








"""     keras.wrapper.scikit_learn import KerasRegressor     
⭐ Read this article to learn about how to use KerasRegressor from the wrapper of scikit
https://stackoverflow.com/questions/77104125/no-module-named-keras-wrappers

"""

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

def buildModel(optimizer, unit, dropout_rate):
    # Initialising the ANN
    model = Sequential()
    
    # Adding the input layer and the first hidden layer
    model.add(Dense(unit, activation = 'relu', input_shape=(X_train.shape[1],)))
    # Adding the second hidden layer
    model.add(Dropout(dropout_rate))
    model.add(Dense(units = 32, activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units = 8, activation = 'relu'))
    # Adding the output layer
    model.add(Dense(units = 1, activation = 'linear'))
    # optimizer.learning_rate.assign(lr)
    # Compiling the ANN
    model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    
    return model

def runGridSearchCV():
    regressor = KerasRegressor(build_fn = buildModel)

    #What hyperparameter we want to play with
    parameters = {
        'batch_size': [16, 32, 64, 128],
        'epochs': [100, 150],
        'optimizer': ['adam', 'rmsprop'],
        # 'lr': [0.001, 0.01, 0.1],
        'unit': [50, 70, 120],
        'dropout_rate': [0.1, 0.2, 0.3]
    }
    grid_search = GridSearchCV(estimator = regressor,
                            param_grid = parameters,
                            scoring = 'neg_mean_absolute_error',
                            cv = 5)

    grid_search = grid_search.fit(X_train, y_train, verbose = 0)

    best_parameters = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters: " + str(best_parameters))




















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





















"""  NOT Great code    """
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