import sys
import runpy
import argparse

def _build_module_name(model_type, algorithm, dataset):
    lt = model_type.lower()
    alg = algorithm.lower()
    ds = dataset

    if lt == 'cova':
        return f"models.COVA.COVA_{alg}_{ds}"
    elif lt == 'retrain':
        return f"models.retrain.retrain_{alg}_{ds}_bpr"
    elif lt == 'original':
        return f"models.original.original_{alg}_{ds}_bpr"
    else:
        raise ValueError(f"Unknown model_type: {model_type}. (choose from original, retrain, cova)")

def _dict_to_argv(args_dict):
    """
    Convert a dictionary to CLI-style arguments.

    Example:
    {'attack': '0.01', 'dataset': 'Yelp'}
        -> ['--attack', '0.01', '--dataset', 'Yelp']

    This allows the target module's argparse to parse arguments as if
    they were passed from the command line.
    """
    argv = []
    if not args_dict:
        return argv
    for k, v in args_dict.items():
        key = f"--{k.replace('_','-')}"
        argv.append(key)
        if isinstance(v, (list, tuple)):
            argv.extend(map(str, v))
        elif isinstance(v, bool):
            # Add as a flag if True; skip if False
            if v:
                # handled as standalone flag (no value appended)
                pass
        else:
            argv.append(str(v))
    return argv

def run_model(model_type, algorithm, dataset, args):
    """
    Preset (dict) mode:
    Build the target module path based on type/algorithm/dataset,
    then run the module as if it were executed as __main__ using runpy.
    The args dict is converted into CLI arguments and passed to the target's argparse.
    """
    module_name = _build_module_name(model_type, algorithm, dataset)
    print(f"--- execute module as __main__ : {module_name} ---")

    # Pass arguments to the target module
    sys.argv = [""] + _dict_to_argv(args)

    try:
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)
    except ModuleNotFoundError:
        print(f"Error: Module '{module_name}' not found. Please check the path or __init__.py.")
    except SystemExit as e:
        # argparse may call sys.exit (e.g., --help)
        print(f"SystemExit: {e}")
    except Exception as e:
        print(f"Error during module execution: {e}")

def run_module_cli(model_type, algorithm, dataset, passthrough_argv):
    """
    CLI mode:
    main.py parses model_type/algorithm/dataset, then forwards
    all remaining arguments directly to the target module's argparse.
    """
    module_name = _build_module_name(model_type, algorithm, dataset)
    print(f"--- execute module as __main__ : {module_name} ---")
    print(f"--- forwarded argv to target: {passthrough_argv} ---")

    # Forward passthrough args directly
    sys.argv = [""] + passthrough_argv

    try:
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)
    except ModuleNotFoundError:
        print(f"Error: Module '{module_name}' not found. Please check the path or __init__.py.")
    except SystemExit as e:
        print(f"SystemExit: {e}")
    except Exception as e:
        print(f"Error during module execution: {e}")

if __name__ == '__main__':
    # ─────────────────────────────────────────────────────────────────────
    # 1) CLI mode (default):
    #    - main.py parses model/module, forwards remaining args to target
    #    - Example:
    #      python main.py --model-type cova --algorithm MF --dataset Amazon_Book --attack 0.01
    #
    # 2) Preset mode (--use-preset flag):
    #    - Executes predefined run_model() calls inside this file
    # ─────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Dispatcher for running target recommender modules.")
    parser.add_argument('--model-type', type=str, default='cova',
                        choices=['original', 'retrain', 'cova'],
                        help="Model family (original/retrain/cova).")
    parser.add_argument('--algorithm', type=str, default='MF',
                        choices=['MF', 'LightGCN'],
                        help="Algorithm name.")
    parser.add_argument('--dataset', type=str, default='Yelp',
                        help="Dataset name (case-sensitive, e.g., Amazon_Book).")
    parser.add_argument('--use-preset', action='store_true',
                        help="Run hard-coded presets instead of CLI passthrough.")

    # Remaining args will be forwarded to the target module
    args, passthrough = parser.parse_known_args()

    if args.use_preset or not passthrough:
        # ───── Preset Mode (default if no extra args) ─────
        # LightGCN (Yelp)
        # run_model('Original', 'LightGCN', 'yelp', {'attack': '0.01'})
        # run_model('Retrain',  'LightGCN', 'yelp', {'attack': '0.01'})
        run_model('COVA', 'LightGCN', 'yelp', {'attack': '0.01'})

        # MF (Amazon_Book)
        # run_model('Original', 'MF', 'Amazon_Book', {'attack': '0.01'})
        # run_model('Retrain',  'MF', 'Amazon_Book', {'attack': '0.01'})
        # run_model('COVA', 'MF', 'Amazon_Book', {'attack': '0.01'})
    else:
        # ───── CLI mode ─────
        # Example:
        # python main.py --model-type cova --algorithm MF --dataset Amazon_Book --attack 0.01 --gcn_layers 2
        run_module_cli(args.model_type, args.algorithm, args.dataset, passthrough)
