#!/usr/bin/env python3
"""
Kaggle Notebook: Train InworldTTS (SFT Stage)
=============================================
This notebook sets up the environment, authenticates with Hugging Face,
runs data vectorization, and starts the Supervised Fine-Tuning (SFT) 
for the InworldTTS model using the Mozilla Common Voice data.

Prerequisites on Kaggle:
  1. Add your Hugging Face Token in Add-ons -> Secrets (Label: HF_TOKEN)
     Note: Make sure your HF account has access to "meta-llama/Llama-3.2-3B-Instruct"!
  2. Add the dataset output from our "Download MCV Thai" notebook.
     (File -> Add dataset -> Your Work -> Select the MCV notebook output)
  3. Enable GPU (Accelerator: GPU T4 x2 or P100)
  4. Enable Internet Access (Required for downloading pretrained models)
"""

import os
import subprocess
import sys

# ==============================================================================
# Step 1: Hugging Face Authentication
# ==============================================================================
print("=" * 60)
print("Step 1: Authenticating with Hugging Face")
print("=" * 60)

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    
    # Export token to environment so scripts (like transformers/huggingface_hub) can use it
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    
    # Install huggingface_hub just in case, then login via Python API
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=True)
    
    print("✅ Hugging Face authentication successful!")
    
except Exception as e:
    print("❌ ERROR: Failed to get HF_TOKEN from Kaggle Secrets.")
    print("Make sure you added 'HF_TOKEN' in the Add-ons -> Secrets menu!")
    print(f"Details: {e}")
    sys.exit(1)

# ==============================================================================
# Step 2: Clone Repository & Install Dependencies
# ==============================================================================
print("\n" + "=" * 60)
print("Step 2: Cloning InworldTTS and Installing Dependencies")
print("=" * 60)

KAGGLE_WORKING = "/kaggle/working"
REPO_DIR = os.path.join(KAGGLE_WORKING, "Inworldtts")

if not os.path.exists(REPO_DIR):
    print("Cloning repository Inworldtts...")
    # Get GitHub token from Kaggle Secrets if available
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        gh_token = user_secrets.get_secret("GH_TOKEN")
        github_repo_url = f"https://oauth2:{gh_token}@github.com/chalitbkb/Inworldtts.git"
    except Exception:
        print("⚠️ GitHub token (GH_TOKEN) not found in Kaggle Secrets. Using public clone URL.")
        github_repo_url = "https://github.com/chalitbkb/Inworldtts.git"
        
    subprocess.run(["git", "clone", github_repo_url, REPO_DIR], check=True)

# Change working directory to the repo so scripts run correctly
os.chdir(REPO_DIR)
print(f"Current working directory: {os.getcwd()}")

print("Installing requirements (this might take a few minutes)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "pythainlp"], check=True) # For Thai Normalizer

print("✅ Dependencies installed successfully!")

# ==============================================================================
# Step 3: Configure Dataset Path
# ==============================================================================
print("\n" + "=" * 60)
print("Step 3: Configuring Dataset")
print("=" * 60)

# Kaggle mounts inputs usually in /kaggle/input/<dataset-slug>/...
# Since this script assumes you added the output from our download script as a dataset,
# we need to find the train.jsonl file dynamically.

input_dir = "/kaggle/input"
train_jsonl_path = ""
output_zip = ""

import zipfile

# 1. Search for _output_.zip first (Kaggle zips large outputs)
for root, dirs, files in os.walk(input_dir):
    if "_output_.zip" in files:
        output_zip = os.path.join(root, "_output_.zip")
        break

if output_zip:
    print(f"📦 Found zipped dataset at: {output_zip}")
    print("⏳ Extracting dataset to /kaggle/working ... (this may take a minute)")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall("/kaggle/working/")
    print("✅ Extraction complete!")
    train_jsonl_path = "/kaggle/working/dataset/train.jsonl"
else:
    # 2. If no zip, search for train.jsonl directly
    for root, dirs, files in os.walk(input_dir):
        if "train.jsonl" in files:
            train_jsonl_path = os.path.join(root, "train.jsonl")
            break

if not train_jsonl_path or not os.path.exists(train_jsonl_path):
    print("⚠️ WARNING: Could not find train.jsonl in /kaggle/input/ or /kaggle/working/")
    print("\n--- DIAGNOSTIC: Here is what Kaggle sees inside /kaggle/input/ ---")
    for root, dirs, files in os.walk(input_dir):
        print(f"Directory: {root}/")
        for f in files[:10]:
            print(f"  - {f}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
    print("---------------------------------------------------------------")
    print("Did you forget to add the MCV output dataset to this notebook, or did it fail to generate?")
    print("For now, creating a dummy dataset path string.")
    train_jsonl_path = "/path/to/your/train.jsonl"
else:
    print(f"✅ Found dataset at: {train_jsonl_path}")

# ==============================================================================
# Step 3.5: Download Codec Checkpoint
# ==============================================================================
print("\n" + "=" * 60)
print("Step 3.5: Downloading Codec Model (HKUSTAudio/xcodec2)")
print("=" * 60)

from huggingface_hub import hf_hub_download
# The data_vectorizer script expects the path directly pointing to a .pt or .ckpt file.
codec_path = hf_hub_download(repo_id="HKUSTAudio/xcodec2", filename="ckpt/epoch=4-step=1400000.ckpt")
print(f"✅ Downloaded codec to: {codec_path}")

# ==============================================================================
# Step 4: Run Data Vectorization
# ==============================================================================
print("\n" + "=" * 60)
print("Step 4: Running Data Vectorization (Encoder processing)")
print("=" * 60)

vectorized_dir = os.path.join(KAGGLE_WORKING, "vectorized_mcv_thai")

if os.path.exists(os.path.join(vectorized_dir, "train_codes.npy")):
    print("⏭️ DING DING DING! Vectorized data already exists!")
    print("Skipping Step 4 to save you 1 hour of waiting. Jumping straight to training!")
else:
    # Per README, data_vectorizer uses torchrun
    cmd_vectorize = [
        "torchrun", "--nproc_per_node", "1", "tools/data/data_vectorizer.py", 
        "--dataset_path", train_jsonl_path,
        "--codec_model_path", codec_path, # Provide the exact downloaded path
        "--output_dir", vectorized_dir,
        "--batch_size", "4"
    ]
    print("Running:", " ".join(cmd_vectorize))
    subprocess.run(cmd_vectorize, check=True)

    # Step 4.5: Merge Shards
    print("Merging Vectorized Shards...")
    cmd_merge = [
        sys.executable, "tools/data/data_merger.py",
        "--dataset_path", vectorized_dir,
        "--remove_shards"
    ]
    subprocess.run(cmd_merge, check=True)
    print("✅ Vectorization and Merging completed")

# ==============================================================================
# Step 5: Start SFT Training
# ==============================================================================
print("\n" + "=" * 60)
print("Step 5: Starting Supervised Fine-Tuning (SFT)")
print("=" * 60)

import json
sft_config_path = "example/configs/sft.json"

print(f"Dynamically updating {sft_config_path} to use our vectorized dataset...")
with open(sft_config_path, "r") as f:
    config = json.load(f)

import torch
import multiprocessing
import psutil

config["train_weighted_datasets"] = { vectorized_dir: 1.0 }
config["val_weighted_datasets"] = { vectorized_dir: 1.0 }
config["training"]["logging_steps"] = 10
config["training"]["eval_steps"] = 50
config["checkpointing"]["keep_only_last_n_checkpoints"] = 5
config["training"]["strategy"] = "ddp"

# ==============================================================================
# 🧠 INTELLIGENT HARDWARE AUTO-CONFIGURATION
# ==============================================================================
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
cpu_cores = multiprocessing.cpu_count()
ram_gb = psutil.virtual_memory().total / (1024**3)

# Safely get properties of the first GPU (assuming homogenous setup like Kaggle)
if torch.cuda.is_available():
    gpu_props = torch.cuda.get_device_properties(0)
    vram_gb = gpu_props.total_memory / (1024**3)
    gpu_name = gpu_props.name.lower()
else:
    vram_gb = 0
    gpu_name = "cpu"

print(f"\n🖥️ Hardware Auto-Detection:")
print(f"   - GPUs: {num_gpus}x {gpu_name.upper()} ({vram_gb:.1f} GB VRAM each)")
print(f"   - CPUs: {cpu_cores} cores")
print(f"   - RAM : {ram_gb:.1f} GB System Memory")

# 1. Precision & Speed Optimization (Architecture specific)
if "t4" in gpu_name or "v100" in gpu_name or "p100" in gpu_name:
    config["training"]["precision"] = "16-mixed" # Turing/Volta/Pascal prefer FP16
    print("   ✓ Selected 16-mixed precision (Optimal for older arch)")
else:
    config["training"]["precision"] = "bf16-mixed" # Ampere/Hopper (A100, H100, 3090, 4090) prefer BF16
    print("   ✓ Selected bf16-mixed precision (Optimal for modern arch)")

# 2. VRAM & Model Capacity Management (The OOM prevention logic)
# Llama-3B needs ~6GB just for base weights in bf16/fp16. 
# LoRA adds memory. Activations add memory based on batch size.

# Very High VRAM (A100 80GB, A6000 48GB) - Run wild!
if vram_gb >= 40:
    config["training"]["batch_size"] = 16
    config["training"]["gradient_accumulation_steps"] = 1
    config["training"]["gradient_checkpointing"] = False # Save time, we have the ram
    lora_r = 64
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# High VRAM (RTX 3090/4090 24GB, A100 40GB) - Comfortable
elif vram_gb >= 22:
    config["training"]["batch_size"] = 8
    config["training"]["gradient_accumulation_steps"] = 2
    config["training"]["gradient_checkpointing"] = True
    lora_r = 32
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Medium VRAM (Kaggle T4 16GB, RTX 4080 16GB) - Tight fit
elif vram_gb >= 14:
    config["training"]["batch_size"] = 1
    config["training"]["gradient_accumulation_steps"] = max(1, int(16 / num_gpus)) # Aim for effective batch of ~16
    config["training"]["gradient_checkpointing"] = True
    lora_r = 16
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Low VRAM (12GB or less) - Survival mode
else:
    config["training"]["batch_size"] = 1
    config["training"]["gradient_accumulation_steps"] = max(1, int(16 / num_gpus))
    config["training"]["gradient_checkpointing"] = True
    lora_r = 8
    target_modules = ["q_proj", "v_proj"] # Minimum required for learning
    print("   ⚠️ WARNING: Low VRAM detected. Running in survival mode (aggressive accumulation, minimal LoRA).")

# 3. CPU Data Loading Optimization
config["training"]["num_workers"] = min(cpu_cores, num_gpus * 4, 8) # Don't overwhelm CPUs

print("\n⚙️ Auto-Configured Training Parameters:")
print(f"   - Batch Size: {config['training']['batch_size']} (per GPU)")
print(f"   - Grad Accum: {config['training']['gradient_accumulation_steps']} steps")
print(f"   - CPU Workers: {config['training']['num_workers']}")
print(f"   - LoRA Rank: {lora_r} | Targets: {len(target_modules)} modules")
print(f"   - Grad Checkpoint: {config['training']['gradient_checkpointing']}")

# Inject LoRA config
config["lora"] = {
    "task_type": "CAUSAL_LM",
    "r": lora_r,
    "lora_alpha": lora_r * 2,
    "target_modules": target_modules,
    "lora_dropout": 0.05,
    "bias": "none"
}

# Force disable NeMo text normalization to prevent unsupported language errors (like 'ja' or 'th')
if "modeling" in config and "parameters" in config["modeling"]:
    config["modeling"]["parameters"]["enable_text_normalization"] = False

with open(sft_config_path, "w") as f:
    json.dump(config, f, indent=4)

print("Remember: Kaggle shuts down after 12 hours. The script will save checkpoints to /kaggle/working/experiments")

cmd_train = [
    "fabric", "run", f"--devices={num_gpus}", "tts/training/main.py",
    "--config_path", "example/configs/sft.json"
]
print("\nRunning:", " ".join(cmd_train))
subprocess.run(cmd_train, check=True)

print("✅ Training logic completed")

# ==============================================================================
# Step 6: Convert Checkpoint for Serving
# ==============================================================================
print("\n" + "=" * 60)
print("Step 6: Converting Checkpoint for Serving")
print("=" * 60)

import glob

experiments_dir = "experiments"
checkpoints = glob.glob(f"{experiments_dir}/*/final_model.pt")

if not checkpoints:
    print("⚠️ No final_model.pt found in experiments directory.")
    print("If it timed out, you may need to find the latest epoch checkpoint and convert manually.")
else:
    # Get the latest if there are multiple runs
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"Found checkpoint: {latest_checkpoint}")
    
    # Save the serving model directly back to Kaggle Working Directory for easy download
    serving_path = os.path.join(KAGGLE_WORKING, "ready_to_serving.pt")
    
    cmd_convert = [
        sys.executable, "tools/serving/convert_checkpoint.py",
        "--checkpoint_path", latest_checkpoint,
        "--output_path", serving_path
    ]
    print("Running:", " ".join(cmd_convert))
    subprocess.run(cmd_convert, check=True)
    
    print(f"✅ Conversion complete! Model ready for Inference: {serving_path}")

print("\n" + "=" * 60)
print("🎉 Notebook Complete! You can now download `ready_to_serving.pt`")
print("=" * 60)
