import argparse
import json
import traceback
from pathlib import Path

from matplotlib import pyplot as plt

import spatial_factorization
from spatial_factorization.utils import misc, preprocess, training, postprocess, visualize
from spatial_factorization import SpatialFactorization

import anndata as ad
from tensorflow_probability import math as tp_math
tf_kernels = tp_math.psd_kernels
import tensorflow as tf


import mlflow

def main():
    parser = argparse.ArgumentParser(description='Run NSF on specified dataset and device.')
    parser.add_argument('--L', type=int, required=True, default=10, help="Number of metagenes to use for all replicates.")
    parser.add_argument('--output_path', type=str, required=True, help="Path at which to save NSF output.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to input dataset.")
    parser.add_argument('--device', type=str, required=True, help="keyword args to use of PyTorch tensors during training.")
    parser.add_argument('--random_state', type=int, help="seed for reproducibility of randomized computations. Default ``0``")

    args = parser.parse_args()
    filtered_args = {key: value for key, value in vars(args).items() if value is not None}

    print(filtered_args)

    output_path = filtered_args.pop("output_path")
    output_path = Path(output_path)
    
    device_index = filtered_args.pop("device")

    with mlflow.start_run():
        trackable_hyperparameters = (
            'L',
            'random_state',
            'dataset_path'
        )
                                            
        mlflow.log_params({
            **{hyperparameter: filtered_args[hyperparameter] for hyperparameter in trackable_hyperparameters if hyperparameter in filtered_args},
        })
        dataset_path  = filtered_args.pop("dataset_path")
        try:
            dataset = ad.read_h5ad(dataset_path)
            num_spots, num_genes = dataset.X.shape

            devices = tf.config.list_physical_devices('GPU')
            gpu = devices[int(device_index)]
            
            tf.config.set_visible_devices(gpu, 'GPU')
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
            )
                                                                                                   
            training_data, _  = preprocess.anndata_to_train_val(                                   
                dataset,                                                                           
                train_frac=1.0,                                                                    
                flip_yaxis=False                                                                   
            )                                                                                      
                                                                                                   
            #convert to tensorflow objects                                                         
            training_tensors = preprocess.prepare_datasets_tf(training_data)                       
            nsf_parameters = {                                                                     
                'J': num_genes, # number of genes                                                  
                'L': filtered_args["L"], # number of spatial metagenes                             
                'Z': training_data["X"], # spatial coordinates                                    
                'random_state': filtered_args['random_state'],
                'nonneg': True,                                                                    
                'psd_kernel': tf_kernels.MaternThreeHalves,                                        
                'lik': 'poi', # Likelihood model                                                   
            }                                                                                      
                                                                                                   
            fit = SpatialFactorization(**nsf_parameters)                                           
            fit.init_loadings(training_data["Y"], X=training_data["X"], sz=training_data["sz"], shrinkage=0.3)
            tro = training.ModelTrainer(fit)                                                       
                                                                                                   
            tro.train_model(*training_tensors, status_freq=50)  

            results = postprocess.interpret_nsf(fit, training_data["X"], S=10000, lda_mode=False)  
            embeddings = results["factors"]                                                        
            metagenes = results["loadings"]                                                        
                                                                                                   
            dataset_name = dataset.obs["batch"].unique()[0]

            dataset.obsm["X"] = embeddings   
            dataset.uns["M"] = {
                dataset_name: metagenes
            }

 
            dataset.write_h5ad(output_path)

            mlflow.log_artifact(output_path)
            if Path(f"./output_gpu:{device_index}.txt").is_file():
                mlflow.log_artifact(f"output_gpu:{device_index}.txt")

        except Exception as e:
            with open(Path(f"./output_gpu:{device_index}.txt"), 'a') as f:
                tb = traceback.format_exc()
                f.write(f'\n{tb}')
        finally:
            if Path(f"./output_gpu:{device_index}.txt").is_file():
                mlflow.log_artifact(f"output_gpu:{device_index}.txt")

if __name__ == "__main__":
    main()
