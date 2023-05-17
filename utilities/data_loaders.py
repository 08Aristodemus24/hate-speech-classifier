import pandas as pd

def load_data(path='', dataset='hate-offensive-speech'):

    if dataset == "hate-offensive-speech":
        df = pd.read_csv(path, index_col=0)
        return df.loc[:, ['tweet', 'class']]
    
    elif dataset == "slur-corpus":
        df = pd.read_csv(path, index_col=0)
        df.index.name = None
        df.reset_index(drop=True, inplace=True)
        return df.loc[:, ['body', 'gold_label']]
    
    elif dataset == "ethos":
        return 0
    
    elif dataset == "reddit":
        return 0 

    
