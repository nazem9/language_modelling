# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import glob
import tiktoken
from tqdm import tqdm
import pytorch_lightning as pl

# Import the lightning module
from lightning_module import LanguageModelLightning

###############################################################################
# Tokenizer setup (using tiktoken)
###############################################################################
enc = tiktoken.get_encoding("o200k_base")
def encode(text: str):
    return enc.encode(text)

def decode(ids: list[int]):
    return enc.decode(ids)

###############################################################################
# Dataset / Data Loading
###############################################################################
class ShardedTextDataset(Dataset):
    def __init__(self, shard_dir):
        """
        Expects shard_dir containing multiple text shard files.
        """
        super().__init__()
        self.txt_files = sorted(glob.glob(f"{shard_dir}/wiki_books_test_*.txt"))

    def __len__(self):
        return len(self.txt_files)

    def load_shard(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return torch.tensor(encode(text), dtype=torch.long)

    def __getitem__(self, idx):
        """
        For demonstration, returns the full shard as a single example. 
        You will likely refine this approach for large data. 
        """
        file_path = self.txt_files[idx]
        return self.load_shard(file_path)

def pad_collate_fn(batch, block_size=256):
    """
    Collate function to slice/pad tokens to block_size and create (input, target).
    """
    # batch is a list of 1D tensors, each from a different shard
    inputs, targets = [], []
    for tokens in batch:
        # If there's not enough tokens, skip or do zero padding
        if len(tokens) < block_size + 1:
            continue
        # Random starting point
        start = random.randint(0, len(tokens) - block_size - 1)
        inp = tokens[start:start+block_size]
        tgt = tokens[start+1:start+block_size+1]
        inputs.append(inp)
        targets.append(tgt)
    if not inputs:
        # Fallback to the first item
        tokens = batch[0]
        tokens = F.pad(tokens, (0, block_size - len(tokens)), "constant", 0)
        inp = tokens[:block_size]
        tgt = tokens[1:block_size+1]
        inputs.append(inp)
        targets.append(tgt)
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)
    return inputs, targets

###############################################################################
# Training
###############################################################################
if __name__ == "__main__":
    shard_dir = "/path/to/arwiki_books_shards/content/sharded"  # Adjust path
    dataset = ShardedTextDataset(shard_dir)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=pad_collate_fn)

    # Initialize the Lightning module
    vocab_size = enc.n_vocab
    steps = 5000
    model_module = LanguageModelLightning(vocab_size=vocab_size, lr=1e-4, steps=steps)

    # Set up the trainer
    trainer = pl.Trainer(
        max_steps=steps,
        gpus=1 if torch.cuda.is_available() else 0,
        gradient_clip_val=1.0
    )

    # Train!
    trainer.fit(model_module, train_loader)