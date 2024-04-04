from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# Load the model and scaler from the pickle files
MODEL = tf.keras.models.load_model('model/')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    

origins = [
    "https://trading-backend-vyzs.onrender.com",  # replace with the origin of your frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    data: list

@app.post("/predict/")
async def predict(data: InputData):
    input_array = np.array(data.data)
    n_steps = 100
    temp_input = input_array.tolist()  # Initialize temp_input with the input data provided by the user

    lst_output = []
    i = 0
    while i < 30:
        if len(temp_input) > 100:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = MODEL.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i += 1
        else:
            x_input = np.array(temp_input)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = MODEL.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i += 1
    # Inverse transform the predicted values before returning them
    response = scaler.inverse_transform(np.array(lst_output)).flatten().tolist()
    return {"prediction": response}
