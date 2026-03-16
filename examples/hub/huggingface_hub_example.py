# coding=utf-8
"""
Hugging Face Hub Integration Examples for Soniq.

This file demonstrates how to use Soniq's Hugging Face Hub integration
for uploading and downloading models.
"""

# =============================================================================
# 1. Basic Usage - Upload and Download Models
# =============================================================================

from soniq import HuggingFaceHub
from soniq.models.vocoders import HifiGAN
from soniq.config.base_config import BaseConfig

# -----------------------------------------------------------------------------
# 1.1 Login to Hugging Face Hub
# -----------------------------------------------------------------------------

def example_login():
    """Example: Login to Hugging Face Hub."""
    hub = HuggingFaceHub()

    # Method 1: Login with token
    result = hub.login(token="hf_xxx")  # Replace with your token
    if result["success"]:
        print(f"Logged in as: {result['username']}")

    # Method 2: Check login status
    status = hub.check_auth()
    print(f"Login status: {status}")

    # Method 3: Logout
    # hub.logout()


# -----------------------------------------------------------------------------
# 1.2 Upload a Trained Model
# -----------------------------------------------------------------------------

def example_upload_model():
    """Example: Upload a model to Hugging Face Hub."""
    hub = HuggingFaceHub()
    hub.login(token="hf_xxx")  # Replace with your token

    # Upload a trained model directory
    url = hub.upload_model(
        model_path="./checkpoints/hifigan",  # Local model directory
        repo_id="myuser/my-hifigan-model",   # Your Hugging Face repo
        model_type="vocoder",                 # Model type
        private=False,                        # Public or private repo
        commit_message="Upload HifiGAN model",
    )
    print(f"Model uploaded to: {url}")


# -----------------------------------------------------------------------------
# 1.3 Download a Model from Hub
# -----------------------------------------------------------------------------

def example_download_model():
    """Example: Download a model from Hugging Face Hub."""
    hub = HuggingFaceHub()

    # Download to cache directory
    model_path = hub.download_model(
        repo_id="soniq/hifigan-base",
    )
    print(f"Model cached at: {model_path}")

    # Download to specific directory
    model_path = hub.download_model(
        repo_id="soniq/hifigan-base",
        local_dir="./models/hifigan",
    )
    print(f"Model downloaded to: {model_path}")


# -----------------------------------------------------------------------------
# 1.4 Load Model Directly from Hub
# -----------------------------------------------------------------------------

def example_load_from_hub():
    """Example: Load a model directly from Hugging Face Hub."""
    from soniq.models.vocoders import HifiGAN

    # Load from Hugging Face Hub using the model's from_pretrained method
    model = HifiGAN.from_pretrained("soniq/hifigan-base")

    # Use the model
    model.eval()
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Number of parameters: {model.num_parameters:,}")


# =============================================================================
# 2. Advanced Usage
# =============================================================================

# -----------------------------------------------------------------------------
# 2.1 Save and Upload in One Step
# -----------------------------------------------------------------------------

def example_save_and_upload():
    """Example: Save a model and upload to Hub in one step."""
    from soniq.models.vocoders import HifiGAN
    from soniq.models.vocoders.hifigan.configuration_hifigan import HifiGANConfig

    # Create or load your model
    config = HifiGANConfig()
    model = HifiGAN(config)

    hub = HuggingFaceHub()
    hub.login(token="hf_xxx")

    # Save and upload in one call
    url = hub.save_and_upload(
        model=model,
        repo_id="myuser/my-hifigan-model",
        config=config,
        model_type="vocoder",
        model_card_kwargs={
            "description": "A high-quality HifiGAN vocoder",
            "language": ["en", "zh"],
            "sample_rate": 24000,
        },
    )
    print(f"Model uploaded to: {url}")


# -----------------------------------------------------------------------------
# 2.2 List Your Models
# -----------------------------------------------------------------------------

def example_list_models():
    """Example: List your models on Hugging Face Hub."""
    hub = HuggingFaceHub()
    hub.login(token="hf_xxx")

    # List all your models
    models = hub.list_models(author="myuser", limit=10)

    for model in models:
        print(f"- {model['id']} (likes: {model['likes']}, downloads: {model['downloads']})")

    # Search for models
    models = hub.list_models(search="hifigan", limit=10)
    print(f"Found {len(models)} models matching 'hifigan'")


# -----------------------------------------------------------------------------
# 2.3 Get Model Information
# -----------------------------------------------------------------------------

def example_get_model_info():
    """Example: Get detailed model information."""
    hub = HuggingFaceHub()

    info = hub.get_model_info("soniq/hifigan-base")

    print(f"Repo ID: {info.repo_id}")
    print(f"Author: {info.author}")
    print(f"Likes: {info.likes}")
    print(f"Downloads: {info.downloads}")
    print(f"Tags: {', '.join(info.tags)}")
    print(f"Files: {info.files}")


# -----------------------------------------------------------------------------
# 2.4 Upload Individual Files
# -----------------------------------------------------------------------------

def example_upload_file():
    """Example: Upload an individual file to Hub."""
    hub = HuggingFaceHub()
    hub.login(token="hf_xxx")

    url = hub.upload_file(
        file_path="./model.bin",
        repo_id="myuser/my-hifigan-model",
        path_in_repo="weights/model.bin",
        commit_message="Upload model weights",
    )
    print(f"File uploaded to: {url}")


# -----------------------------------------------------------------------------
# 2.5 Download Individual Files
# -----------------------------------------------------------------------------

def example_download_file():
    """Example: Download an individual file from Hub."""
    hub = HuggingFaceHub()

    file_path = hub.download_file(
        repo_id="soniq/hifigan-base",
        filename="config.json",
        local_dir="./configs",
    )
    print(f"Config downloaded to: {file_path}")


# =============================================================================
# 3. Using Model's Built-in push_to_hub Method
# =============================================================================

def example_push_to_hub():
    """Example: Use the model's built-in push_to_hub method."""
    from soniq.models.vocoders import HifiGAN

    # Load or create your model
    model = HifiGAN.from_pretrained("./checkpoints/hifigan")

    # Push directly to Hub
    url = model.push_to_hub(
        repo_id="myuser/my-hifigan-model",
        model_type="vocoder",
        private=False,
    )
    print(f"Model pushed to: {url}")


# =============================================================================
# 4. Command Line Interface
# =============================================================================

"""
The Soniq CLI provides commands for Hub operations:

# Login
soniq-hub login --token hf_xxx

# Check status
soniq-hub status

# Upload a model
soniq-hub upload ./checkpoints/hifigan \
    --repo-id myuser/my-model \
    --model-type vocoder \
    --private

# Download a model
soniq-hub download myuser/my-model \
    --local-dir ./models

# List your models
soniq-hub list --author myuser

# Get model info
soniq-hub info myuser/my-model

# Delete a repository (use with caution)
soniq-hub delete myuser/my-model -y
"""


# =============================================================================
# 5. Complete Training and Upload Workflow
# =============================================================================

def example_training_workflow():
    """Example: Complete training workflow with Hub upload."""
    from soniq import HuggingFaceHub
    from soniq.models.vocoders import HifiGAN
    from soniq.models.vocoders.hifigan.configuration_hifigan import HifiGANConfig

    # Step 1: Setup
    config = HifiGANConfig()
    model = HifiGAN(config)

    # Step 2: Train your model (pseudo-code)
    # trainer = Trainer(model, config, train_data, val_data)
    # trainer.train()

    # Step 3: Save trained model
    model.save_pretrained("./checkpoints/hifigan-final")

    # Step 4: Upload to Hugging Face Hub
    hub = HuggingFaceHub()
    hub.login(token="hf_xxx")

    url = hub.upload_model(
        model_path="./checkpoints/hifigan-final",
        repo_id="myuser/hifigan-final",
        model_type="vocoder",
        model_card_kwargs={
            "description": "High-quality HifiGAN vocoder trained on LibriTTS",
            "sample_rate": 24000,
            "language": ["en"],
        },
    )

    print(f"Training complete! Model available at: {url}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Soniq Hugging Face Hub Integration Examples")
    print("=" * 50)

    # Run examples (uncomment to run)
    # example_login()
    # example_upload_model()
    # example_download_model()
    # example_load_from_hub()
    # example_save_and_upload()
    # example_list_models()
    # example_get_model_info()
    # example_push_to_hub()
    # example_training_workflow()
