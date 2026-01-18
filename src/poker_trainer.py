from transformers import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig

class PokerTrainer:
    def __init__(self, model, tokenizer, train_cfg: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.train_cfg = train_cfg["train"]

    def _set_SFT_arguments(self)-> SFTConfig:
        return SFTConfig(
            output_dir = self.train_cfg["output_dir"],
            per_device_train_batch_size = self.train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps = self.train_cfg["gradient_accumulation_steps"],
            learning_rate = float(self.train_cfg["learning_rate"]),
            warmup_ratio = self.train_cfg["warmup_ratio"],
            max_steps = self.train_cfg["max_steps"],
            bf16 = self.train_cfg["bf16"], fp16 = self.train_cfg["fp16"],
            gradient_checkpointing = self.train_cfg["gradient_checkpointing"],
            lr_scheduler_type = self.train_cfg["lr_scheduler_type"],
            logging_steps = self.train_cfg["logging_steps"],
            save_steps = self.train_cfg["save_steps"], eval_steps = self.train_cfg["eval_steps"],
            save_total_limit = self.train_cfg["save_total_limit"],
            report_to = self.train_cfg["report_to"],
            optim = self.train_cfg["optim"],
            completion_only_loss = self.train_cfg["completion_only_loss"],

            load_best_model_at_end = self.train_cfg["load_best_model_at_end"],
            metric_for_best_model = self.train_cfg["metric_for_best_model"],
            greater_is_better = self.train_cfg["greater_is_better"],
            eval_strategy = self.train_cfg["eval_strategy"], save_strategy = self.train_cfg["save_strategy"],
        )
    def build_trainer(self, train_dataset, eval_dataset, callbacks=None) -> SFTTrainer:
        if callbacks is None:
            callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]

        sft_args = self._set_SFT_arguments()
        return SFTTrainer(
            model = self.model,
            args = sft_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = self.tokenizer,
            callbacks = callbacks,
        )

    def train(self, trainer: SFTTrainer):
        trainer.train()
        trainer.model.save_pretrained()

    def save_adapter(self, trainer: SFTTrainer, save_path: str):
        trainer.model.save_pretrained(save_path)