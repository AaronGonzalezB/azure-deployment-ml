import json
import numpy as np
import os
import pickle
import joblib

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'cluster_mood.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    y_hat = model.predict(data)
    clusters = ['Workout','Concert','Ambience']
    #return y_hat.tolist()
    return json.dumps({'predicted_class': clusters[int(y_hat)]})