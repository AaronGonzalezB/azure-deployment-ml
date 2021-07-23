from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig

if __name__ == "__main__":
    # Connect to Azure ML Workspace
    ws = Workspace.from_config(path='./.azureml', _file_name='config.json')

    # Experiment instance
    experiment = Experiment(workspace=ws, name='day1-experiment-train-sklearn-remote')

    # Setup pytorch enviroment
    config = ScriptRunConfig(source_directory='./src',
                             script='train-remote.py',
                             compute_target='cpu-cluster')

    # set up pytorch environment
    env = Environment.from_conda_specification(
        name='sklearn-entrega-env',
        file_path='./.azureml/sklearn-entrega-remote-env.yml'
    )
    config.run_config.environment = env

    # Execute experiment
    run = experiment.submit(config)

    aml_url = run.get_portal_url()
    print(aml_url)