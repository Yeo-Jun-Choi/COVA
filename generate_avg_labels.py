import numpy as np
import pandas as pd
import os

def generate_avg_labels(data_path, dataset, attack):
    """
    SCIF unlearning  avg_labels.npy  .
    
    Args:
        data_path:  
        dataset:  
        attack: attack 
    """
    #   
    train_random_path = os.path.join(data_path, dataset, attack, 'train_random.csv')
    save_path = os.path.join(data_path, dataset, attack, 'avg_labels.npy')
    
    # train_random.csv  
    train_random = pd.read_csv(train_random_path)
    
    # unlearn  
    n_unlearn = len(train_random)
    
    #     ( 0.5    )
    # Yelp    interaction positive 1.0 
    avg_labels = np.ones(n_unlearn) * 1.0
    
    #  
    np.save(save_path, avg_labels)
    print(f"avg_labels.npy  : {save_path}")
    print(f"  : {len(avg_labels)}")
    print(f"  : {np.mean(avg_labels)}")

if __name__ == '__main__':
    # Yelp   avg_labels 
    generate_avg_labels('Data/Process/', 'Yelp', '0.01')
    
    #    ()
    # generate_avg_labels('Data/Process/', 'Gowalla', '0.01')
    # generate_avg_labels('Data/Process/', 'Amazon_Book', '0.01') 