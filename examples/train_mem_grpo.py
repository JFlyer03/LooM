import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
import swanlab

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom_agent.trainers.mem_grpo_trainer import MemGRPOTrainer, MemGRPOArguments
import loom_agent.envs  # Register envs
import loom_agent.modules # Register modules
import loom_agent.agents # Register agents

def main():
    parser = HfArgumentParser((MemGRPOArguments,))
    # # MemGRPOArguments does not define model_path/data_path fields, 
    # so use a separate dataclass or argparse to handle the model/data paths.
    
    # Let's use argparse for the main script wrapper, and then populate MemGRPOArguments.
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    cli_parser.add_argument("--data_path", type=str, required=True, help="Path to the training data")
    cli_parser.add_argument("--wandb_name", type=str, default=None, help="SwanLab experiment name")
    
    # We parse known args for CLI, and the rest for MemGRPOArguments
    args, remaining_args = cli_parser.parse_known_args()
    
    # Parse MemGRPOArguments from remaining_args
    grpo_args = parser.parse_args_into_dataclasses(args=remaining_args)[0]
    
    print(f"Loading model from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "device_map": "auto",
        "trust_remote_code": True
    }
    
    try:
        import flash_attn
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")
    except ImportError:
        print("Flash Attention 2 not found, using default attention")
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        **model_kwargs
    )
    
    # Init SwanLab
    if args.wandb_name:
        swanlab_api_key = os.getenv("SWANLAB_API_KEY", "YOUR_SWANLAB_API_KEY")
        if swanlab_api_key != "YOUR_SWANLAB_API_KEY":
            os.environ["SWANLAB_API_KEY"] = swanlab_api_key
        else:
            print("Warning: SWANLAB_API_KEY is still using the placeholder value.")
        swanlab.init(
            project="loom",
            config=vars(grpo_args),
            name=args.wandb_name
        )
    
    print("Initializing Trainer...")
    trainer = MemGRPOTrainer(
        model=model,
        args=grpo_args,
        tokenizer=tokenizer
    )
    
    print(f"Starting Training with Agent: {grpo_args.agent_type}, Env: {grpo_args.env_type}...")
    trainer.train(args.data_path)

if __name__ == "__main__":
    main()
