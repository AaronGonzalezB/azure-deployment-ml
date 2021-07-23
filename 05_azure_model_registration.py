# 07-model-registration-azure.py
from azureml.core import Workspace
from azureml.core import Model

if __name__ == "__main__":
    ws = Workspace.from_config(path='./.azureml',_file_name='config.json')      # Conexion al workspace

    model = Model.register(model_name='model',                             # Registro del modelo
                           tags={'area': 'entrega','0':'Workout','1':'Concert','2':'Ambience'},            # tags - etiquetas informativas del modelo
                           model_path='outputs/cluster_mood.pkl',
                           workspace = ws)
    print(model.name, model.id, model.version, sep='\t')