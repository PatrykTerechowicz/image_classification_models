from torch.utils.data import Dataset

class DatasetPreloaded(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self._data = []
        for data in dataset:
            self._data.append(data)
    
    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)