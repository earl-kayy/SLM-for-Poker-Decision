import yaml

from argparse import ArgumentParser
from tqdm import tqdm

from src.data_processor import DataProcessor
from src.model_loader import ModelLoader
from src.poker_trainer import PokerTrainer

parser = ArgumentParser(description="Specify train configuration paths")
parser.add_argument("-a", "--adapterpath", required=True, help="Path to the finetuned model")
parser.add_argument("-c", "--configurationpath", required=True, help="")
args = parser.parse_args()

adapter_path = args.adapterpath
configuration_path = args.configurationpath

with open(configuration_path, "r") as f:
    cfg = yaml.safe_load(f)

train_dataset = DataProcessor.load_preprocessed_dataset(train_path="data/train")

model_loader = ModelLoader(model_cfg=cfg)
model, tokenizer = model_loader.load_to_train()

poker_trainer = PokerTrainer(model=model, tokenizer=tokenizer, train_cfg=cfg)
trainer = poker_trainer.build_trainer(
    train_dataset=train_dataset["train"], 
    eval_dataset=train_dataset["val"]
)
poker_trainer.train(trainer=trainer)

poker_trainer.save_adapter(trainer=trainer, save_path=adapter_path)