from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig

ws = Workspace.from_config(path="./.azureml",_file_name="config.json")      # Conectarse al espacio de trabajo de Azure

experiment = Experiment(workspace=ws, name='day1-experiment-testing-workspace')     # Nombre experimento a ejecutar

config = ScriptRunConfig(source_directory='./src', script = 'test-workspace.py', compute_target='cpu-cluster')      # Encapsulamiento de los componentes que se van a llamar (codigo, ambientes, requerimientos)

run = experiment.submit(config)     # Enviamos el codigo para ejecucion

aml_url = run.get_portal_url()      # Monitoreo de atributos en el portal
print(aml_url)