
import os
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.common import get_base_dir

def create_dummy_checkpoint():
    config = GPTConfig(
        n_layer=2,
        n_head=1,
        n_kv_head=1,
        n_embd=128,
        vocab_size=50257,
        sequence_len=64
    )
    model = GPT(config)
    
    # Save checkpoint
    base_dir = get_base_dir()
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", "dummy")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    save_checkpoint(
        checkpoint_dir,
        0,
        model.state_dict(),
        None,
        {
            "step": 0,
            "val_bpb": 10.0,
            "model_config": config.__dict__,
            "user_config": {},
            "device_batch_size": 1,
            "max_seq_len": 64
        }
    )
    print(f"Created dummy checkpoint at {checkpoint_dir}")

if __name__ == "__main__":
    create_dummy_checkpoint()
