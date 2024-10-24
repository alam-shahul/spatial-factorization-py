import textwrap
from pathlib import Path

def generate_mlproject_file(project_name: str, output_filepath: str = "MLproject"):
    """Generate MLproject file for command line use.

    Args:
        project_name: name of project

    """

    mlproject_contents = f"""
    name: {project_name}

    entry_points:
      train:
        parameters:
          dataset_path: {{type: str, default: "input.h5ad"}}
          output_path: {{type: str, default: "output.h5ad"}} 
          L: {{type: int}}
          device: {{type: int}}
          random_state: {{type: int, default: 0}} 
        command: "spatial-factorization-mlflow --dataset_path={{dataset_path}} --output_path={{output_path}}
                                     --L={{L}}
                                     --device={{device}}
                                     --random_state={{random_state}} > output_gpu:{{device}}.txt 2>&1"
      
      spatial-factorization:
        parameters:
          configuration_filepath: path
        command: "spatial-factorization-grid-search --configuration_filepath {{configuration_filepath}}"
    """

    with open(output_filepath, "w") as f:
        f.writelines(textwrap.dedent(mlproject_contents).strip())

def load_from_mlflow_run(run, client, file_suffix="_result", **loading_kwargs):
    """Load trained model from MLflow run.
    
    Args:
        run: MLflow run containing model artifact.
        client: MLflow client initialized to point to experiment location.
        loading_kwargs: kwargs that can be passed to Popari pretrained initialization.
    
    """
    run_id  = run.info.run_id
    artifacts = client.list_artifacts(run_id)

    trained_model_artifact, = list(filter(lambda artifact: Path(artifact.path).stem.endswith(file_suffix), artifacts))

    print(run.data.params)
    _, base_uri = run.info.artifact_uri.split(":")
    trained_model_filepath = Path(base_uri) / trained_model_artifact.path

    trained_model = load_trained_model(trained_model_filepath, **loading_kwargs)
    
    return trained_model

