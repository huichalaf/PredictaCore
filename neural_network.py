import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

async def load_data(input_name = "input.csv", output_name = "output.csv"):

    input_data = pd.read_csv(input_name)
    output_data = pd.read_csv(output_name)
    return input_data, output_data

async def create_model(X, Y, model_name = "model", layers = 15, neurons = 20, epochs = 500, batch_size = 10):
    
    model = Sequential()
    for i in range(layers):
        model.add(Dense(neurons, input_dim=X.shape[1], activation='relu'))  # Capa oculta
    
    model.add(Dense(1, activation='linear'))  # Capa de salida
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(X.shape[1])
    model.fit(X, Y, epochs=epochs, batch_size=10)
    model.save_weights("weights/"+model_name+"_weights.h5")
    model.save("weights/"+model_name+"_model.h5")

    return model

async def predict_(model, X):
    # Convertimos los valores de X en float
    print(X)
    #tomamos solo la primera fila del dataframe X
    X = X.iloc[0]
    X = np.array(X, dtype=float)
    # Reshape X to have two dimensions
    X = X.reshape(1, -1)
    print(X)
    Y = model.predict(X)
    return Y.tolist()  # Convertir ndarray a lista