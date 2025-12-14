
import os
import json
import torch
import argparse
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model
from nanochat.inner_learner import InnerLearner
from contextlib import nullcontext

def run_demo(args):
    # Setup
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    
    # Load prompts
    demos = []
    with open('demos/prompts.jsonl', 'r') as f:
        for line in f:
            demos.append(json.loads(line))
            
    results = {}
    
    # Run comparison: Baseline vs Inner Learner
    configs = [
        {'name': 'baseline', 'inner_learn': False},
        {'name': 'inner_learn', 'inner_learn': True}
    ]
    
    for config in configs:
        print(f"\nRunning {config['name']}...")
        
        # Load model anew for strictly fair comparison (and to reset state)
        model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag)
        
        if config['inner_learn']:
            inner_learner = InnerLearner(
                model.config, 
                device, 
                dtype=ptdtype,
                lr=args.inner_lr,
                decay=args.inner_decay,
                clip=args.inner_clip
            )
            model.inner_learner = inner_learner
            print(f"Inner Learner attached: lr={args.inner_lr}")
        else:
            model.inner_learner = None
            
        engine = Engine(model, tokenizer)
        
        # Special tokens
        # Safe getter for special tokens (since we might be using vanilla GPT2 tokenizer)
        def get_special(t, default):
            tid = tokenizer.encode_special(t)
            return tid if tid is not None else default
            
        bos = get_special("<|bos|>", 50256)
        user_start = get_special("<|user_start|>", 50257) 
        user_end = get_special("<|user_end|>", 50257)
        assistant_start = get_special("<|assistant_start|>", 50257)
        assistant_end = get_special("<|assistant_end|>", 50256)
        
        results[config['name']] = []
        
        for demo in demos:
            print(f"  Task: {demo['task']}")
            
            # Reset inner learner for new conversation if enabled
            if model.inner_learner:
                model.inner_learner.reset()
                
            conversation_tokens = [bos]
            demo_output = {'task': demo['task'], 'turns': []}
            
            for turn_idx, user_input in enumerate(demo['turns']):
                conversation_tokens.append(user_start)
                conversation_tokens.extend(tokenizer.encode(user_input))
                conversation_tokens.append(user_end)
                
                conversation_tokens.append(assistant_start)
                
                # Generate
                # For demo, strict greedy generation or low temp to see effects clearly?
                # User suggested temp 0.8.
                generate_kwargs = {
                    "num_samples": 1,
                    "max_tokens": 100, # limit output length
                    "temperature": 0.8,
                    "top_k": 50,
                }
                
                response_tokens = []
                with autocast_ctx:
                    for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
                        token = token_column[0]
                        response_tokens.append(token)
                
                # Ensure end token
                if response_tokens and response_tokens[-1] != assistant_end:
                    response_tokens.append(assistant_end)
                
                conversation_tokens.extend(response_tokens)
                
                response_text = tokenizer.decode(response_tokens).replace("<|assistant_end|>", "")
                print(f"    Turn {turn_idx+1}: {response_text[:50]}...")
                demo_output['turns'].append({'input': user_input, 'output': response_text})
                
            results[config['name']].append(demo_output)
            
    # Save results
    with open('demos/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to demos/results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', type=str, default="base", help="Source of the model: base|sft|mid|rl")
    parser.add_argument('-g', '--model-tag', type=str, default=None)
    parser.add_argument('--device-type', type=str, default='')
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16')
    
    # Inner learner defaults from user request
    parser.add_argument('--inner-lr', type=float, default=0.01)
    parser.add_argument('--inner-decay', type=float, default=0.999)
    parser.add_argument('--inner-clip', type=float, default=0.5)
    
    args = parser.parse_args()
    run_demo(args)
