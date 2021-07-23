from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace

interactive_auth = InteractiveLoginAuthentication(tenant_id='XXXXXX')   # Set the tenant ID
ws = Workspace.create(name='azure-ml-entrega',
                        subscription_id='XXXXXX',   # Set 
                        resource_group='rg_machine_learning_entrega',
                        create_resource_group=True,
                        location='eastus2',
                        auth=interactive_auth
                        )

ws.write_config(path='.azureml')