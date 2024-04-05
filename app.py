from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
app = FastAPI()
from sklearn.preprocessing import MinMaxScaler
import uvicorn
# Load the model and scaler from the pickle files
MODEL = tf.keras.models.load_model('model/')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update this with your frontend URL
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    data: list

@app.post("/predict/")
async def predict(data: InputData = Body):
    print("Data Received",data)
    input_array = np.array(data.data[0])
    print("Data Received",input_array)
    
    scaled_input = scaler.fit_transform(input_array.reshape(-1, 1))
    print("Scaled Input",scaled_input)
    n_steps = 100
    temp_input = scaled_input.flatten().tolist()  
    print("Temp Input",temp_input)
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
    print("output",lst_output)
    response = scaler.inverse_transform(np.array(lst_output)).flatten().tolist()
    print("response",response)
    return {"prediction": response}
if(__name__) == '__main__':
        uvicorn.run(
        "app:app",
        host    = "0.0.0.0",
        port    = 10000, 
        reload  = True
    )