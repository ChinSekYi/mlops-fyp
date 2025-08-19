def save_object(file_path, obj):

    """
    Save an object to a file using pickle.
    """
    import os, pickle
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_model(x, y):
    
    report= {
            'accuracy': accuracy_score(x, y),
            'precision': precision_score(x, y),
            'recall': recall_score(x, y),
            'f1': f1_score(x, y)
        }
    return report



def load_object(file_path):
    """
    Load an object from a file using pickle.
    """
    import pickle
    with open(file_path, "rb") as file_obj:
        return pickle.load(file_obj)

