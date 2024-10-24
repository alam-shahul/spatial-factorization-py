from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import itertools
import toml

import numpy as np

import mlflow
import mlflow.tracking
import mlflow.projects
from mlflow.tracking import MlflowClient

from pathlib import Path
from spatial_factorization.mlflow.util import generate_mlproject_file

def run():
    parser = argparse.ArgumentParser(description='Run NSF hyperparameter grid search.')
    parser.add_argument('--configuration_filepath', type=str, required=True, help="Path to configuration file.")
    parser.add_argument('--mlflow_output_path', type=str, default=".", help="Where to output MLflow results.")
    parser.add_argument('--debug', action='store_true', help="Whether to dump print statements to console or to file.")

    args = parser.parse_args()

    with open(args.configuration_filepath, "r") as f:
        configuration = toml.load(f)

    mlflow_output_path = Path(args.mlflow_output_path)
    generate_mlproject_file(configuration['runtime']['project_name'], mlflow_output_path / "MLproject")

    hyperparameter_names = [
        'L',
        'random_state',
    ]

    if "dataset_paths" in configuration['runtime']:
        dataset_paths = configuration['runtime']['dataset_paths']
    elif "dataset_path" in configuration['runtime']:
        dataset_paths = [configuration['runtime']['dataset_path']]

    configuration['hyperparameters']['dataset_path'] = dataset_paths

    # Fix this up so that you can input a list of CUDA devices instead of a number of processes
    if "device_list" in configuration['runtime']:
        device_list = configuration['runtime']['device_list']
        num_processes = len(device_list)
    elif "num_processes" in configuration['runtime']:
        num_processes = configuration['runtime']['num_processes']
        device_list = [f'{device_number}' for device_number in range(num_processes)]

    tracking_client = MlflowClient()

    device_status = [False for _ in range(num_processes)]
    def generate_evaluate_function(parent_run, experiment_id):
        """Generates function to evaluate NSF.

        """

        def evaluate(params):
            """Start parallel NSF run and track progress.


            """
            
            device_index = device_status.index(False)
            device = device_list[device_index]
            device_status[device_index] = True 

            child_run = tracking_client.create_run(experiment_id, tags={"mlflow.parentRunId": parent_run.info.run_id})
            # with mlflow.start_run(nested=True) as child_run:
            p = mlflow.projects.run(
                run_id=child_run.info.run_id,
                uri=str(mlflow_output_path),
                entry_point="train_debug" if args.debug else "train",
                parameters={
                    **{parameter_name: params[parameter_name] for parameter_name in params if params[parameter_name] is not None },
                    "output_path": f"./device_{device}_result.h5ad",
                    "device": device,
                },
                env_manager="local",
                experiment_id=experiment_id,
                synchronous=False,
            )
            succeeded = p.wait()
            device_status[device_index] = False

            tracking_client.set_terminated(child_run.info.run_id)

            if succeeded:
                training_run = tracking_client.get_run(p.run_id)
                metrics = training_run.data.metrics

                # else:
                #     raise RuntimeError("A run failed during initialization. This likely points "
                #         "to an improperly formatted grid search configuration file.")
            else:
                tracking_client.set_terminated(p.run_id, "FAILED")

            return p.run_id

        return evaluate

    with mlflow.start_run() as parent_run:
        experiment_id = parent_run.info.experiment_id

        hyperparameter_options_list = []
        for hyperparameter_name in hyperparameter_names:
            if hyperparameter_name not in configuration['hyperparameters']:
                default_options = [null_hyperparameters[hyperparameter_name]]
                hyperparameter_options_list.append(default_options)
                continue
            
            search_space = configuration['hyperparameters'][hyperparameter_name]

            start = search_space['start']
            end = search_space['end']
            scale = search_space['scale']
            gridpoints = search_space['gridpoints']
            dtype = search_space['dtype']

            if scale == 'log':
                gridspace = np.logspace
            elif scale == 'linear':
                gridspace = np.linspace
            
            hyperparameter_options = gridspace(start, end, num=gridpoints)

            if dtype == 'int':
                hyperparameter_options = np.rint(hyperparameter_options).astype(int)

            hyperparameter_options_list.append(hyperparameter_options)

        hyperparameter_names.append('dataset_path')
        hyperparameter_options_list.append(dataset_paths)

        options = list(dict(zip(hyperparameter_names, hyperparameter_choice)) for hyperparameter_choice in itertools.product(*hyperparameter_options_list))

        print(f"Number of grid search candidates: {len(options)}")

        evaluate = generate_evaluate_function(parent_run, experiment_id)
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            _ = executor.map(evaluate, options,)

        # find the best run, log its metrics as the final metrics of this run.
        client = MlflowClient()
        runs = client.search_runs(
            [experiment_id], f"tags.mlflow.parentRunId = '{parent_run.info.run_id}' "
        )

if __name__ == "__main__":
    run()

