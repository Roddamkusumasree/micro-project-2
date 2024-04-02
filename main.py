# Import necessary libraries
from azureml.core import Workspace, Experiment, Environment
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.train.automl import AutoMLConfig
# Load Azure ML workspace
ws = Workspace.from_config()
# Create or attach an Azure ML compute cluster
cluster_name = "aml-cluster"
try:
 compute_target = ComputeTarget(workspace=ws, name=cluster_name)
 print('Found existing compute target.')
except ComputeTargetException:
 print('Creating a new compute target...')
 compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS12_V2', 
max_nodes=4)
 compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
 compute_target.wait_for_completion(show_output=True)
# Create a new experiment
experiment_name = 'energy-consumption-prediction'
experiment = Experiment(ws, experiment_name)
# Create a run configuration
run_config = RunConfiguration()
run_config.environment.python.user_managed_dependencies = False
run_config.environment.python.conda_dependencies = 
CondaDependencies.create(conda_packages=['numpy','pandas','scikit-learn'])
# Define AutoML configuration
automl_config = AutoMLConfig(
 task='regression',
 primary_metric='normalized_root_mean_squared_error',
 experiment_timeout_hours=1,
 training_data=training_data,
 label_column_name='energy_consumption',
 compute_target=compute_target,
 enable_early_stopping=True,
 featurization='auto'
)
# Submit the AutoML experiment
run = experiment.submit(automl_config)
run.wait_for_completion(show_output=True)s 
# Retrieve the best model
best_run, fitted_model = run.get_output()
# Perform optimization based on predictions using the best mode