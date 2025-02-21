# LM

Welcome! This repository contains a minimal transformer-based language modeling pipeline in PyTorch. It uses [PyTorch Lightning](https://www.pytorchlightning.ai/) for training and [tiktoken](https://github.com/openai/tiktoken) for tokenization.

## Files

1. **model.py**  
   Contains the definitions of:  
   • Head (self-attention head)  
   • MultiHeadAttention (collection of attention heads)  
   • FeedForward (position-wise feedforward network)  
   • Block (transformer block)  
   • LM (the main transformer-based language model)

2. **lightning_module.py**  
   Wraps the LM model into a PyTorch Lightning module (LanguageModelLightning).  
   Manages training steps, validation steps, and optimizer/lr_scheduler creation.

3. **train.py**  
   Provides the following:  
   • Dataset and DataLoader setup (ShardedTextDataset and a custom collate function).  
   • Example usage of the DataLoader, model creation, and training routine with PyTorch Lightning.  
   • The code loads text from multiple shard files, makes a train set, and trains the transformer model.

## Requirements

• Python 3.8+ recommended  
• PyTorch >= 1.10  
• PyTorch Lightning >= 1.9  
• tiktoken >= 0.4.0 (to handle tokenization)  
• tqdm (for progress bars)

You may install these dependencies as follows:

```bash
pip install torch pytorch-lightning tiktoken tqdm
```

If you plan to use a GPU, ensure that you have a compatible PyTorch CUDA installation.

## Usage

1. Clone this repository or copy the files to your project.
2. Adjust the shard directory path (and optionally other hyperparameters) in train.py.  
   That is, in train.py, modify this line accordingly:
   ```python
   shard_dir = "/path/to/arwiki_books_shards/content/sharded"
   ```
3. Run the training script:
   ```bash
   python train.py
   ```
   By default, it will look for shard files named like "wiki_books_test_*.txt" in the specified directory. These files should contain plain text.

During training, PyTorch Lightning will create default checkpoints in the same folder (unless otherwise configured). To customize, see [PyTorch Lightning docs](https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html).

## Tokenization

We use tiktoken (specifically "o200k_base") for both tokenizing input text and for decoding predictions. To change the tokenizer, adjust the lines in train.py that instantiate and use the tokenizer:
```python
enc = tiktoken.get_encoding("o200k_base")
def encode(text: str):
    return enc.encode(text)

def decode(ids: list[int]):
    return enc.decode(ids)
```

If you are using a different vocabulary or encoding, replace it here.

## Lightning Module

In lightning_module.py, our LanguageModelLightning class handles:  
• Forward pass & loss calculation  
• Optimizer and scheduler setup  
• Training and validation steps  

This modular approach allows you to reuse the LM model in other contexts or frameworks more easily.

## Model Architecture

The language model (LM) includes:
- Multi-head self-attention blocks (transformer blocks)  
- Feedforward sublayers connected by residual connections  
- Token embeddings and positional embeddings  

For text generation (auto-regressive prediction), see the `generate` method in LM. After a model is trained, you can do something like:
```python
from model import LM
import torch

# Suppose 'model' is a trained LM instance on GPU
prompt_ids = torch.tensor([[0]], dtype=torch.long).to('cuda')  # 0 is usually <BOS> or any prompt token
generated = model.generate(prompt_ids, max_new_tokens=50)
print(generated)
```
Then decode from token IDs to text using the decode function.

## License

This project does not contain explicit licensing information. Please adapt or license according to your needs, and be mindful of dependencies with their respective licenses.

## Contributing

If you find any issue or have suggestions, feel free to open an issue or a PR in your own fork.

## Contact

For questions or discussions, open an issue or reach out to the maintainers of this repository. We hope you find this example helpful for building and experimenting with transformers on text data. Enjoy!