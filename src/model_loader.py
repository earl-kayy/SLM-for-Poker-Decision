import torch

from peft import LoraConfig, get_peft_model, PeftModel
from peft.utils.other import prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class ModelLoader:
    def __init__(self, model_cfg: dict):
        self.quant_cfg = model_cfg["quant"]
        self.base_model = model_cfg["base_model"]
        self.lora_cfg = model_cfg["lora"]
        self.model = None

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def _build_bnb_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=self.quant_cfg["load_in_4bit"],
            bnb_4bit_quant_type = self.quant_cfg["quant_type"],
            bnb_4bit_compute_dtype = torch.bfloat16,
            bnb_4bit_use_double_quant = self.quant_cfg["double_quant"],
        )

    def _load_base_model(self):
        bnb = self._build_bnb_config()
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config = bnb,
            device_map = "auto",
        )
        return model

    def _prepare_model(self, model):
        model = prepare_model_for_kbit_training(model)
        return model

    def _build_lora_config(self) -> LoraConfig:
        lora_cfg = LoraConfig(
            r = self.lora_cfg["r"],
            lora_alpha = self.lora_cfg["alpha"],
            target_modules = self.lora_cfg["target_modules"],
            lora_dropout = self.lora_cfg["dropout"],
            bias = self.lora_cfg["bias"],
            task_type = self.lora_cfg["task_type"]
        )
        return lora_cfg

    def _attach_lora(self, model, lora_cfg: LoraConfig):
        model = get_peft_model(model, lora_cfg)
        return model
        
    def _load_finetuned_model(self, model, adapter_path: str):
        model = PeftModel.from_pretrained(model, adapter_path)
        return model
    
    def load_to_train(self):
        tokenizer = self._load_tokenizer()

        base_model = self._load_base_model()        
        model = self._prepare_model(base_model)
        
        lora_cfg = self._build_lora_config()
        model = self._attach_lora(model, lora_cfg)
        return model, tokenizer

    def load_to_test(self, adapter_path: str):
        tokenizer = self._load_tokenizer()
        tokenizer.padding_side = "left"

        base_model = self._load_base_model()

        fine_tuned_model = self._load_finetuned_model(base_model, adapter_path)
        return fine_tuned_model.eval(), tokenizer
    