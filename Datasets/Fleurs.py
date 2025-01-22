from datasets import load_dataset, Dataset

class FleursdatasetTest(Dataset):

    """
    -- Downloading fleurs dataset (french language speaking subset)
    """
    def __init__(self):
        
        fleurs_asr = load_dataset("google/fleurs", "fr_fr")
        
        # see structure
        self.fleurs_dataset_test = fleurs_asr["test"]


    def __getitem__(self, idx):

        return self.fleurs_dataset_test[idx]["audio"]['array'],  self.fleurs_dataset_test[idx]["transcription"], self.fleurs_dataset_test[idx]["audio"]['sampling_rate']

    def __len__(self):
        return len(self.fleurs_dataset_test)
    