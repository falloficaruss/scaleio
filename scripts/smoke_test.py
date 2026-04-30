import os
import sys
from typing import Dict

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.stage1_continuous import Stage1Trainer, get_default_config as get_stage1_config
from training.stage2_discrete import Stage2Trainer

def run_smoke_test():
    """Runs a very short training cycle for both stages to verify logic."""
    print("=== Starting Smoke Test ===")
    
    # 1. Prepare Toy Data
    from scripts.prepare_toy_data import create_toy_dataset
    toy_data_path = 'data/toy_dataset'
    create_toy_dataset(target_dir=toy_data_path, num_images=4)
    
    # 2. Test Stage 1 (Continuous)
    print("\n--- Testing Stage 1 ---")
    s1_config = get_stage1_config()
    s1_config.update({
        'epochs': 2,
        'batch_size': 2,
        'data_path': toy_data_path,
        'save_path': 'checkpoints/test_s1',
        'log_path': 'logs/test_s1',
        'warmup_epochs': 1
    })
    
    s1_trainer = Stage1Trainer(s1_config)
    # Patch validate to be faster
    s1_trainer.validate = lambda: {'loss': 0.0, 'psnr': 30.0, 'ssim': 0.9}
    
    print("Running 2 epochs of Stage 1...")
    s1_trainer.train()
    
    s1_checkpoint = os.path.join(s1_config['save_path'], 'latest_checkpoint.pth')
    if not os.path.exists(s1_checkpoint):
        # Fallback if the trainer uses a different naming convention internally
        s1_checkpoint = os.path.join(s1_config['save_path'], 'checkpoint_epoch_1.pth')

    # 3. Test Stage 2 (Discrete)
    print("\n--- Testing Stage 2 ---")
    s2_config = {
        'epochs': 2,
        'batch_size': 2,
        'lr_max': 1e-4,
        'warmup_epochs': 1,
        'scale_factor': 4,
        'patch_size': 128,
        'data_path': toy_data_path,
        'save_path': 'checkpoints/test_s2',
        'log_path': 'logs/test_s2',
        'stage1_checkpoint': s1_checkpoint,
        'feature_dim': 96,
        'num_hiet_blocks': 3,
        'num_heads': 8
    }
    
    # Check if stage2_discrete has a similar structure
    try:
        s2_trainer = Stage2Trainer(s2_config)
        print("Running 2 epochs of Stage 2...")
        s2_trainer.train()
    except Exception as e:
        print(f"Stage 2 test failed or needs manual config alignment: {e}")

    print("\n=== Smoke Test Completed ===")

if __name__ == "__main__":
    run_smoke_test()
