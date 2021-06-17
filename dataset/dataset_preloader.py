import tqdm
from torch.utils.data import Dataset

class DatasetPreloaded(Dataset):
    """
        
    """
    def __init__(self, dataset, verbose=1):
        super().__init__()
        self._data = []
        dataset_iter = dataset
        if verbose==1: dataset_iter = tqdm.tqdm(dataset, total=len(dataset))
        for data in dataset_iter:
            self._data.append(data)
    
    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)