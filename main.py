import importlib
import argparse

def run_model(model_type, algorithm, dataset, args):
    """
    Constructs the module path based on the given type, algorithm, and dataset, and executes it.
    """
    
    # 1. Construct Module Path
    if model_type.lower() == 'cova':
        # COVA 모델 경로: models/COVA/COVA_[algorithm]_[dataset].py
        module_name = f"models.COVA.COVA_{algorithm.lower()}_{dataset.lower()}"
    elif model_type.lower() == 'retrain':
        # Retrain 모델 경로: models/retrain/retrain_[algorithm]_[dataset]_bpr.py
        module_name = f"models.retrain.retrain_{algorithm.lower()}_{dataset.lower()}_bpr"
    elif model_type.lower() == 'original':
        # Original 모델 경로: models/original/original_[algorithm]_[dataset]_bpr.py
        module_name = f"models.original.original_{algorithm.lower()}_{dataset.lower()}_bpr"
    else:
        raise ValueError(f"알 수 없는 model_type: {model_type}. (original, retrain, cova 중 선택)")

    print(f"--- execute module : {module_name} ---")

    try:
        # 2. Dynamic Module Load
        model_module = importlib.import_module(module_name)
        
        # 3. Argument Handling (as the original script might use argparse)
        # Note: This block currently defines an argparse object but doesn't fully use it
        # to process the 'args' dictionary into a Namespace object before calling 'main'.
        # The current implementation assumes 'main' in the target script accepts a dictionary 
        # or similar structure that can be derived from the 'args' dictionary.
        
        # Create an argparse object (to mimic the original script's behavior)
        parser = argparse.ArgumentParser()
        
        # Add common arguments (assuming most scripts use them)
        parser.add_argument('--attack', type=float, default=0.01)
        parser.add_argument('--dataset', type=str, default='Yelp')
        
        # Arguments needed for LightGCN
        if algorithm.lower() == 'lightgcn':
            parser.add_argument('--gcn_layers', type=int, default=1)
            
        # The 'args' dictionary passed to this function already contains the specific parameters.
        # This code assumes the target script's main function is structured to accept the 'args' dict.
        
        # *** Assuming the target script's main function takes a dictionary as argument: ***
        
        model_module.main(args)
        
        # *** If all scripts call main() without arguments and rely on sys.argv:
        # You would need to temporarily manipulate sys.argv here (more complex).
        
        
    except ModuleNotFoundError:
        print(f"Error: Module '{module_name}' not found. Please check the path.")
    except AttributeError:
        print(f"Error: Module '{module_name}' does not have a 'main' function defined.")
    except Exception as e:
        print(f"Error during module execution: {e}")



if __name__ == '__main__':
    
    # Define settings based on the required execution examples.
    
    # ----------------------------------------------------
    # LightGCN (Yelp) Execution Settings
    # ----------------------------------------------------
    
    # Original LightGCN
    #run_model('Original', 'LightGCN', 'Yelp', {'attack': '0.01', 'gcn_layers': '1'})

    # Retraining LightGCN
    # run_model('Retrain', 'LightGCN', 'Yelp', {'attack': '0.01', 'gcn_layers': '1'})
    
    # COVA LightGCN
    # run_model('COVA', 'LightGCN', 'Yelp', {'attack': '0.01', 'dataset': 'Yelp', 'gcn_layers': '1'})


    # ----------------------------------------------------
    # MF (Yelp) Execution Settings
    # ----------------------------------------------------
    
    # Original MF
    run_model('Original', 'MF', 'Yelp', {'attack': '0.01'})
    
    # Retraining MF
    # run_model('Retrain', 'MF', 'Yelp', {'attack': '0.01'})
    
    # COVA MF
    # run_model('COVA', 'MF', 'Yelp', {'attack': '0.01', 'dataset': 'Yelp'})
    
    # Uncomment the configuration you wish to run.