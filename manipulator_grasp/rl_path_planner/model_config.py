"""
RL Model Configuration
Centralized configuration for trained RL models

This file allows you to easily switch between different trained models
by updating the paths or using environment variables.
"""

import os

# Base directory for RL models
_BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(_BASE_DIR, 'models')

# ==================== PLACE PHASE MODEL ====================
# Model for place phase (with object attachment)
# Update this to point to your best trained model

PLACE_PHASE_CONFIG = {
    # Option 1: Use best_model.zip (recommended after training)
    # 'model_path': os.path.join(MODELS_DIR, 'place_with_object_v1', 'best_model.zip'),
    # 'vecnormalize_path': os.path.join(MODELS_DIR, 'place_with_object_v1', 'best_model_vecnormalize.pkl'),
    
    # Option 2: Use final_model.zip
    'model_path': os.path.join(MODELS_DIR, 'place_with_object_v1', 'final_model.zip'),
    'vecnormalize_path': os.path.join(MODELS_DIR, 'place_with_object_v1', 'final_model_vecnormalize.pkl'),
    
    # Option 3: Use specific checkpoint
    # 'model_path': os.path.join(MODELS_DIR, 'place_with_object_v1', 'place_phase_2500000_steps.zip'),
    # 'vecnormalize_path': os.path.join(MODELS_DIR, 'place_with_object_v1', 'place_phase_vecnormalize_2500000_steps.pkl'),
    
    # Training configuration (for reference)
    'drop_zone_center': [0.6, 0.2, 0.83],  # Target position used during training
    'success_threshold': 0.10,  # 10cm threshold
    'max_steps': 500,
}

# ==================== APPROACH PHASE MODEL (Legacy) ====================
# Original task-space model (currently not used, but kept for reference)

APPROACH_PHASE_CONFIG = {
    'model_path': os.path.join(MODELS_DIR, 'task_space_v5_8_collision_check', 'best_model.zip'),
    'vecnormalize_path': os.path.join(MODELS_DIR, 'task_space_v5_8_collision_check', 'final_model_vecnormalize.pkl'),
    'drop_zone_center': [0.6, 0.2, 0.83],
    'success_threshold': 0.10,
    'max_steps': 200,
}

# ==================== ENVIRONMENT VARIABLE OVERRIDES ====================
# You can override paths using environment variables:
#   export RL_PLACE_MODEL=/path/to/your/model.zip
#   export RL_PLACE_VECNORM=/path/to/your/vecnormalize.pkl

def get_place_phase_config():
    """Get place phase model configuration with env var overrides."""
    config = PLACE_PHASE_CONFIG.copy()
    
    # Check for environment variable overrides
    if 'RL_PLACE_MODEL' in os.environ:
        config['model_path'] = os.environ['RL_PLACE_MODEL']
        print(f"[Config] Using RL_PLACE_MODEL from environment: {config['model_path']}")
    
    if 'RL_PLACE_VECNORM' in os.environ:
        config['vecnormalize_path'] = os.environ['RL_PLACE_VECNORM']
        print(f"[Config] Using RL_PLACE_VECNORM from environment: {config['vecnormalize_path']}")
    
    # Auto-detect if paths don't exist
    if not os.path.exists(config['model_path']):
        # Try to find in logs directory instead
        logs_dir = os.path.join(MODELS_DIR, '..', 'logs')
        if os.path.exists(logs_dir):
            # Search for place_with_object models
            for dirname in os.listdir(logs_dir):
                if 'place' in dirname.lower():
                    candidate_model = os.path.join(logs_dir, dirname, 'best_model.zip')
                    if os.path.exists(candidate_model):
                        print(f"[Config] Auto-detected model in logs: {candidate_model}")
                        config['model_path'] = candidate_model
                        config['vecnormalize_path'] = candidate_model.replace('.zip', '_vecnormalize.pkl')
                        break
    
    return config


def get_approach_phase_config():
    """Get approach phase model configuration (legacy)."""
    return APPROACH_PHASE_CONFIG.copy()


# ==================== QUICK ACCESS ====================
# For convenience, expose the configs directly
def get_active_place_model():
    """Returns (model_path, vecnormalize_path) for place phase."""
    config = get_place_phase_config()
    return config['model_path'], config['vecnormalize_path']
