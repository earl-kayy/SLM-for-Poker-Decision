# SLM for Poker Decision Making🃏

A fine-tuning framework for Small Language Models (SLMs) to make strategic poker decisions using QLoRA (Quantized Low-Rank Adaptation) techniques.

## Overview

This project demonstrates how to adapt small language models (1B) for poker decision-making tasks. The pipeline includes training data downloading, model fine-tuning, and comprehensive evaluation on both preflop and postflop poker scenarios.

## Features

- **Quantized Fine-tuning**: Uses 4-bit quantization and LoRA adapters for efficient memory usage
- **Poker-specific Training**: Fine-tunes models on PokerBench dataset for strategic decision-making
- **Comprehensive Evaluation**: Separate evaluation metrics for preflop and postflop decisions
- **Adapter Management**: Stores and loads trained model adapters independently

## Project Structure

```
SLM for Poker Decision/
├── train.py                     # Main training script
├── evaluate.py                  # Evaluation script for trained models
├── download_train_data.py       # Script to download PokerBench dataset (Training Data)
├── requirements.txt             # Python dependencies
├── configs/                     # Training Configuration files (Used for Inference as well to call the same base model)
│   ├── gemini.yaml              # Gemma model configuration
│   ├── llama.yaml               # Llama model configuration
│   └── sample_format.yaml       # Sample format/template of how to write configuration files
├── data/                        # Training and test data
│   ├── train/                   # Training dataset (Need to download using download_train_data.py)
│   └── test/                    # Test datasets
│       ├── preflop_test.json    # Preflop decision test set
│       └── postflop_test.json   # Postflop decision test set
├── src/                         # Source code modules
│   ├── data_processor.py        # Dataset loading and preprocessing
│   ├── model_loader.py          # Model loading with quantization (Load for train/test)
│   ├── poker_trainer.py         # Custom trainer for SFT
│   └── eval.py                  # Evaluation metrics computation
├── adapter_checkpoint/          # Saved model adapters
│   ├── gemma_qlora_adapter/     # Fine-tuned Gemma adapter
│   ├── llama_qlora_adapter/     # Fine-tuned Llama adapter
└── notebooks/                   # Jupyter notebooks for exploration
    ├── FineTuning_Gemma-3-1B-IT_first_try.ipynb
    └── FineTuning_Llama3.2_1B_third_try.ipynb
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/earl-kayy/SLM-for-Poker-Decision.git
cd "SLM for Poker Decision"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Download Training Data

Download the PokerBench training dataset:
```bash
python download_train_data.py
```

This script fetches the poker decision dataset and saves it to `data/train/`.

### 2. Train a Model

Fine-tune a model using an adapter file (where you want to save the fine-tuned adapter) 
and a configuration file (training configuration):

**For Llama 3.2 1B:**
```bash
python train.py \
  --adapterpath adapter_checkpoint/llama_qlora_adapter \
  --configurationpath configs/llama.yaml
```

### 3. Evaluate a Model

Evaluate a fine-tuned model on preflop and postflop test sets:

```bash
python evaluate.py \
  --adapterpath adapter_checkpoint/llama_qlora_adapter \
  --configurationpath configs/llama.yaml
```

This produces:
- Preflop metrics: Model performance on preflop decision scenarios
- Postflop metrics: Model performance on postflop decision scenarios

## Configuration

Configuration files are YAML-based and located in the `configs/` directory.
It specifies how you want to quantize the base model, structure LoRA adapter, and train.
File is also later used for inference to call the same quantized base model.
Sample template is provided in `configs/sample_format.yaml`



## Architecture

### Data Pipeline
1. **Download**: Fetch PokerBench dataset from Hugging Face
2. **Load**: Load train/test data
3. **Preprocess**: Convert to SFT (Supervised Fine-Tuning) format with instruction-response pairs
4. **Split**: 90-10 train-validation split

### Model Pipeline

#### For Fine-tuning
1. **Load Quantized Model**: Initialize base model with 4-bit quantization (BitsAndBytes)
2. **Attach LoRA**: Attach LoRA adapters to the base model
3. **Train**: Supervised fine-tuning using custom PokerTrainer
4. **Save**: Store adapter weights independently inside `adapter_checkpoint` folder

#### For Evaluation (Post Fine-tuning)
1. **Load Quantized Model**: Initialize base model with 4-bit quantization (BitsAndBytes)
2. **Load and Attach Trained LoRA**: Attach Fine-tuned LoRA adapters to the base model
4. **Evaluate**: Evaluate the model

## Results

### Evaluation Metrics

The models are evaluated using two key metrics:

1. **Action Accuracy (AA)**: This metric evaluates whether the model correctly predicts the poker action (fold, check, call, bet, raise, all-in).

2. **Exact Match (EM)**: This metric measures whether the model outputs exactly match the ground truth response, including all action parameters. For example, "bet 100" must match completely—both the action (bet) and the amount (100) must be correct. It's a stricter evaluation criterion than Action Accuracy.

### Model Performance

#### Llama 3.2 1B Instruct
- **Preflop Test Set** :
  - Action Accuracy: 0.9
  - Exact Match: 0.891

- **Postflop Test Set** :
  - Action Accuracy: 0.824
  - Exact Match: 0.819

#### Gemma 3 1B It
- **Preflop Test Set** :
  - Action Accuracy: 0.822
  - Exact Match: 0.816

- **Postflop Test Set** :
  - Action Accuracy: 0.788
  - Exact Match: 0.784

## Technical Details

### Quantization Strategy
- **4-bit Quantization**: Reduces model memory from ~13GB to ~3.5GB for 1B models
- **BitsAndBytes**: Efficient quantization library enabling fine-tuning on consumer GPUs

### Efficiency Techniques
- **LoRA**: Low-Rank Adaptation reduces trainable parameters to ~1% of base model
- **Adapter Storage**: Only adapters are saved (MB-scale), not full model weights

### Data Format
Training data uses the following SFT format:
```
### Instruction:
[Poker decision prompt]

### Response:
[Decision output]
```

## 📂 Dataset & License

This project utilizes the **[PokerBench](https://huggingface.co/datasets/RZ412/PokerBench)** dataset, which is originally released under the **Apache License 2.0**.

**Note:** The full dataset can also be accessed via the original source link provided above.