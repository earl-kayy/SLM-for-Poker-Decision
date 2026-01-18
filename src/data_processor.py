from datasets import DatasetDict, load_from_disk, load_dataset

class DataProcessor:
    @staticmethod
    def _load_raw_data(train_path: str = None, test_preflop_path: str = None, test_postflop_path: str = None) -> DatasetDict:
        if train_path is not None:
            return load_from_disk(train_path)
        
        elif test_preflop_path is not None and test_postflop_path is not None:
            split_files = {
                "preflop": test_preflop_path,
                "postflop": test_postflop_path
            }
            return load_dataset("json", data_files=split_files)
        
        else:
            raise ValueError("Either train_path or both test_preflop_path and test_postflop_path must be provided.")
    
    @staticmethod
    def _to_sft_text(ex):
        text = f"### Instruction:\n{ex['instruction'].strip()}\n\n### Response:\n{ex['output'].strip()}"
        return {"text": text}

    @staticmethod
    def _preprocess_dataset(train_data: DatasetDict = None, test_data: DatasetDict = None) -> DatasetDict:
        if train_data is not None:
            full_train_dataset = train_data.map(
                DataProcessor._to_sft_text, remove_columns=train_data.column_names
            )

            split_dataset = full_train_dataset.train_test_split(test_size=0.1, seed=7)

            train_dataset_proc = DatasetDict({
                "train": split_dataset["train"],
                "val": split_dataset["test"]
            })
            return train_dataset_proc  
        
        elif test_data is not None:
            test_dataset_proc = DatasetDict()
            test_dataset_proc["preflop_test"] = test_data["preflop"].map(
                DataProcessor._to_sft_text, remove_columns=test_data["preflop"].column_names
            )

            test_dataset_proc["postflop_test"] = test_data["postflop"].map(
                DataProcessor._to_sft_text, remove_columns=test_data["postflop"].column_names
            )
            return test_dataset_proc
        
        else:
            raise ValueError("Either train_data or test_data must be provided.")
        
    @staticmethod
    def load_preprocessed_dataset(train_path: str = None, test_preflop_path: str = None, test_postflop_path: str = None) -> DatasetDict:
        if train_path is not None:
            train_data = DataProcessor._load_raw_data(train_path=train_path)
            return DataProcessor._preprocess_dataset(train_data=train_data)
        
        elif test_preflop_path is not None and test_postflop_path is not None:
            test_data = DataProcessor._load_raw_data(test_preflop_path=test_preflop_path, test_postflop_path=test_postflop_path)
            return DataProcessor._preprocess_dataset(test_data=test_data)
        
        else:
            raise ValueError("Either train_path or both test_preflop_path and test_postflop_path must be provided.")