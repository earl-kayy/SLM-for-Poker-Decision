import yaml

from argparse import ArgumentParser

from src.data_processor import DataProcessor
from src.eval import evaluate_on_dataset
from src.model_loader import ModelLoader

parser = ArgumentParser(description="Specify model, tokenizer, and dataset paths")
parser.add_argument("-a", "--adapterpath", required=True, help="Path to the finetuned model")
parser.add_argument("-c", "--configurationpath", required=True, help="")
args = parser.parse_args()

adapter_path = args.adapterpath
configuration_path = args.configurationpath

with open(configuration_path, "r") as f:
    cfg = yaml.safe_load(f)

#
test_data = DataProcessor.load_preprocessed_dataset(test_preflop_path="data/test/preflop_test.json", test_postflop_path="data/test/postflop_test.json")

model_loader = ModelLoader(cfg)
model_loader, tokenizer = model_loader.load_to_test(adapter_path)

metrics = evaluate_on_dataset(model_loader, tokenizer, test_data["preflop_test"], batch_size=8, max_new_tokens=16)
print("Preflop metrics:", metrics)

metrics = evaluate_on_dataset(model_loader, tokenizer, test_data["postflop_test"], batch_size=8, max_new_tokens=16)
print("Postflop metrics:", metrics)