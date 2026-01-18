from datasets import load_dataset

dataset = load_dataset("RZ412/PokerBench")
dataset["train"].save_to_disk("data/train")
# dataset["test"].save_to_disk("data/test")