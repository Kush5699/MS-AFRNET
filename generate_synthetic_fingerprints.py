"""
Synthetic Fingerprint Generator — using AMSL StyleGAN2 Model
═══════════════════════════════════════════════════════════════

Generates fingerprint images from the pre-trained StyleGAN2 generator
(network-snapshot-010400.pkl) bundled with the AMSL SynFP SGR v1 dataset.

Each identity (seed) gets multiple variations via latent space perturbation,
ensuring MIN_IMAGES_PER_CLASS ≥ 5 for ArcFace training.

Output: /kaggle/working/synthetic_fingerprints/
        subjectXXXXX/
            img_00.png
            img_01.png
            ...
            img_09.png

Run this ONCE on Kaggle, save output as a dataset, then attach to training.

Kaggle Inputs Required:
  - kushpatel7391/afr-training-synthetic  (contains the .pkl model)
"""

import os
import sys
import time
import pickle
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Path to the StyleGAN2 pickle model
MODEL_PATH = "/kaggle/input/datasets/kushpatel7391/afr-training-synthetic/AMSL_SynFP_SGR_v1/network-snapshot-010400.pkl"

# Output directory
OUTPUT_DIR = "/kaggle/working/synthetic_fingerprints"

# Generation parameters — BATCH 2 (different seeds from batch 1)
NUM_IDENTITIES = 40000       # 40K identities → 400K images
VARIATIONS_PER_ID = 10       # Images per identity
PERTURBATION_SCALE = 0.12
TRUNCATION_PSI = 0.7
IMAGE_SIZE = 224
BATCH_SIZE = 16
SEED_OFFSET = 200000         # ← DIFFERENT from batch 1 (was 100000)

# Save as JPEG to reduce disk usage
SAVE_FORMAT = "jpg"
JPEG_QUALITY = 80

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════════
#  1. LOAD STYLEGAN2 GENERATOR
# ═══════════════════════════════════════════════════════════════

print(f"\n📦 Loading StyleGAN2 model from: {MODEL_PATH}")

# StyleGAN2 pickles need special handling
# The pickle contains a dict with keys: 'G', 'D', 'G_ema'
# G_ema is the exponential moving average generator (best quality)

# First, try loading with the standard approach
try:
    # StyleGAN2-ADA / official NVIDIA format
    sys.path.insert(0, '/kaggle/working')
    
    # Some StyleGAN2 pickles require the original source code to unpickle
    # We'll use a compatibility wrapper
    class _TFNetworkStub:
        """Stub for TensorFlow network references in the pickle."""
        pass
    
    class _LegacyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Handle legacy TF references
            if 'dnnlib' in module or 'training' in module or 'torch_utils' in module:
                try:
                    return super().find_class(module, name)
                except (ModuleNotFoundError, AttributeError):
                    return _TFNetworkStub
            return super().find_class(module, name)
    
    with open(MODEL_PATH, 'rb') as f:
        try:
            data = pickle.load(f)
        except Exception:
            f.seek(0)
            data = _LegacyUnpickler(f).load()
    
    if isinstance(data, dict):
        if 'G_ema' in data:
            G = data['G_ema']
            print("  ✅ Loaded G_ema (EMA generator)")
        elif 'G' in data:
            G = data['G']
            print("  ✅ Loaded G (generator)")
        else:
            print(f"  Available keys: {list(data.keys())}")
            raise ValueError("No generator found in pickle")
    else:
        G = data
        print("  ✅ Loaded generator directly")
    
    # Move to GPU if possible
    if hasattr(G, 'to'):
        G = G.to(DEVICE)
    if hasattr(G, 'eval'):
        G.eval()
    
    # Detect latent dimension
    if hasattr(G, 'z_dim'):
        Z_DIM = G.z_dim
    elif hasattr(G, 'mapping') and hasattr(G.mapping, 'z_dim'):
        Z_DIM = G.mapping.z_dim
    else:
        Z_DIM = 512  # Default for StyleGAN2
    
    print(f"  Latent dim: {Z_DIM}")
    
    # Detect if it's a PyTorch or TF model
    IS_PYTORCH = hasattr(G, 'parameters')
    print(f"  Framework: {'PyTorch' if IS_PYTORCH else 'Legacy/TF-converted'}")
    
    GENERATOR_LOADED = True

except Exception as e:
    print(f"  ❌ Failed to load StyleGAN2 model: {e}")
    print(f"\n  Falling back to NOISE-BASED synthetic generation...")
    print(f"  (This creates texture-rich fingerprint-like images for training)")
    GENERATOR_LOADED = False
    Z_DIM = 512


# ═══════════════════════════════════════════════════════════════
#  2. GENERATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def generate_stylegan2_batch(G, z_batch, truncation_psi=0.7):
    """Generate images from StyleGAN2 generator."""
    with torch.no_grad():
        # Standard StyleGAN2 forward pass
        if hasattr(G, 'mapping') and hasattr(G, 'synthesis'):
            # Modern StyleGAN2/3 format
            ws = G.mapping(z_batch, None, truncation_psi=truncation_psi)
            imgs = G.synthesis(ws)
        else:
            # Try direct call
            try:
                imgs = G(z_batch, None, truncation_psi=truncation_psi)
            except TypeError:
                imgs = G(z_batch, None)
        
        # Convert from [-1, 1] to [0, 255]
        imgs = (imgs.clamp(-1, 1) + 1) * 127.5
        imgs = imgs.to(torch.uint8)
        
        return imgs


def generate_procedural_fingerprint(seed, size=512):
    """
    Generate fingerprint-like synthetic images with TYPE VARIETY.
    5 types rotate based on seed: optical, latent, contactless, thermal, rolled.
    """
    rng = np.random.RandomState(seed)
    ftype = seed % 5  # 0=optical, 1=latent, 2=contactless, 3=thermal, 4=rolled
    
    x = np.linspace(-np.pi, np.pi, size)
    y = np.linspace(-np.pi, np.pi, size)
    xx, yy = np.meshgrid(x, y)
    
    # Ridge orientation and frequency
    angle = rng.uniform(0, np.pi)
    freq = rng.uniform(4, 8)
    
    # Base ridge pattern
    ridges = np.sin(freq * (xx * np.cos(angle) + yy * np.sin(angle)))
    cx, cy = rng.uniform(-0.5, 0.5, 2)
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    curve_angle = np.arctan2(yy - cy, xx - cx) + rng.uniform(0, 2*np.pi)
    ridges += 0.5 * np.sin(freq * 0.8 * r + curve_angle)
    ridges = ridges / (np.abs(ridges).max() + 1e-8)
    
    noise = rng.randn(size, size).astype(np.float32)
    
    if ftype == 0:
        # OPTICAL: clean, high contrast, circular mask
        img = 0.8 * ridges + 0.2 * noise
        mask_r = rng.uniform(0.35, 0.45) * size
        bg, fg = rng.uniform(210, 240), rng.uniform(20, 60)
    elif ftype == 1:
        # LATENT: faint, noisy, partial coverage, smudged
        img = 0.3 * ridges + 0.7 * noise  # Very noisy
        mask_r = rng.uniform(0.15, 0.30) * size  # Partial
        bg, fg = rng.uniform(180, 220), rng.uniform(100, 160)  # Low contrast
    elif ftype == 2:
        # CONTACTLESS: softer ridges, rectangular mask, slight blur effect
        ridges_soft = ridges * 0.6
        img = 0.65 * ridges_soft + 0.35 * noise
        mask_r = rng.uniform(0.30, 0.42) * size
        bg, fg = rng.uniform(200, 230), rng.uniform(40, 90)
    elif ftype == 3:
        # THERMAL: inverted contrast, streaky
        streak = np.sin(freq * 1.5 * yy) * 0.3
        img = 0.6 * ridges + 0.15 * noise + 0.25 * streak
        mask_r = rng.uniform(0.25, 0.40) * size
        bg, fg = rng.uniform(30, 70), rng.uniform(180, 230)  # INVERTED
    else:
        # ROLLED: full coverage, high detail, elliptical
        img = 0.85 * ridges + 0.15 * noise
        mask_r = rng.uniform(0.40, 0.48) * size  # Large coverage
        bg, fg = rng.uniform(215, 245), rng.uniform(15, 50)
    
    # Apply mask
    mask = np.sqrt((xx / np.pi * size/2)**2 + (yy / np.pi * size/2)**2) < mask_r
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_out = np.full((size, size), bg, dtype=np.float32)
    img_out[mask] = fg + (bg - fg) * img_norm[mask]
    
    return np.clip(img_out, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
#  3. BATCH GENERATION
# ═══════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)

total_images = NUM_IDENTITIES * VARIATIONS_PER_ID
print(f"\n{'='*60}")
print(f"  SYNTHETIC FINGERPRINT GENERATION")
print(f"{'='*60}")
print(f"  Identities:       {NUM_IDENTITIES:,}")
print(f"  Variations/ID:    {VARIATIONS_PER_ID}")
print(f"  Total images:     {total_images:,}")
print(f"  Perturbation:     {PERTURBATION_SCALE}")
print(f"  Output:           {OUTPUT_DIR}")
print(f"  Format:           {SAVE_FORMAT.upper()} (quality={JPEG_QUALITY})")
est_size_gb = total_images * (40 if SAVE_FORMAT == 'jpg' else 250) / 1e9
print(f"  Est. disk usage:  {est_size_gb:.1f} GB")
print(f"{'='*60}\n")

start_time = time.time()
images_saved = 0
last_report = 0

for identity_idx in range(NUM_IDENTITIES):
    seed = SEED_OFFSET + identity_idx
    subject_dir = os.path.join(OUTPUT_DIR, f"subject{seed:07d}")
    os.makedirs(subject_dir, exist_ok=True)
    
    if GENERATOR_LOADED:
        # Generate base latent vector for this identity
        rng = torch.Generator().manual_seed(seed)
        base_z = torch.randn(1, Z_DIM, generator=rng).to(DEVICE)
        
        for var_idx in range(VARIATIONS_PER_ID):
            # Add small perturbation for variation
            var_rng = torch.Generator().manual_seed(seed * 1000 + var_idx)
            perturbation = torch.randn(1, Z_DIM, generator=var_rng).to(DEVICE)
            z = base_z + perturbation * PERTURBATION_SCALE
            
            try:
                img_tensor = generate_stylegan2_batch(G, z, TRUNCATION_PSI)
                
                # Convert to PIL
                img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
                if img_np.shape[2] == 1:
                    img = Image.fromarray(img_np[:, :, 0], mode='L').convert('RGB')
                else:
                    img = Image.fromarray(img_np)
                
            except Exception as e:
                if var_idx == 0 and identity_idx == 0:
                    print(f"  ⚠️ StyleGAN2 forward failed: {e}")
                    print(f"  Falling back to procedural generation...")
                    GENERATOR_LOADED = False
                # Fallback for this image
                img_arr = generate_procedural_fingerprint(seed * 100 + var_idx)
                img = Image.fromarray(img_arr, mode='L').convert('RGB')
            
            # Resize if needed
            if img.size != (IMAGE_SIZE, IMAGE_SIZE):
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            
            # Save
            fname = f"img_{var_idx:02d}.{SAVE_FORMAT}"
            fpath = os.path.join(subject_dir, fname)
            if SAVE_FORMAT == 'jpg':
                img.save(fpath, quality=JPEG_QUALITY)
            else:
                img.save(fpath)
            
            images_saved += 1
    
    else:
        # Procedural generation fallback
        for var_idx in range(VARIATIONS_PER_ID):
            img_arr = generate_procedural_fingerprint(seed * 100 + var_idx)
            img = Image.fromarray(img_arr, mode='L').convert('RGB')
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            
            fname = f"img_{var_idx:02d}.{SAVE_FORMAT}"
            fpath = os.path.join(subject_dir, fname)
            if SAVE_FORMAT == 'jpg':
                img.save(fpath, quality=JPEG_QUALITY)
            else:
                img.save(fpath)
            
            images_saved += 1
    
    # Progress report every 1000 identities
    if (identity_idx + 1) % 1000 == 0 or identity_idx == NUM_IDENTITIES - 1:
        elapsed = time.time() - start_time
        rate = images_saved / elapsed
        eta = (total_images - images_saved) / rate if rate > 0 else 0
        pct = images_saved / total_images * 100
        print(f"  [{identity_idx+1:,}/{NUM_IDENTITIES:,}] "
              f"{images_saved:,}/{total_images:,} images ({pct:.1f}%) | "
              f"{rate:.0f} img/s | ETA: {eta/60:.1f} min")


# ═══════════════════════════════════════════════════════════════
#  4. SUMMARY
# ═══════════════════════════════════════════════════════════════

elapsed = time.time() - start_time

# Count actual files
actual_count = 0
actual_subjects = 0
for d in os.listdir(OUTPUT_DIR):
    dp = os.path.join(OUTPUT_DIR, d)
    if os.path.isdir(dp):
        actual_subjects += 1
        actual_count += len([f for f in os.listdir(dp) if f.endswith(('.jpg', '.png'))])

# Disk usage
total_size = 0
for root, dirs, files in os.walk(OUTPUT_DIR):
    for f in files:
        total_size += os.path.getsize(os.path.join(root, f))

print(f"\n{'='*60}")
print(f"  ✅ GENERATION COMPLETE")
print(f"{'='*60}")
print(f"  Subjects created: {actual_subjects:,}")
print(f"  Images created:   {actual_count:,}")
print(f"  Disk usage:       {total_size / 1e9:.2f} GB")
print(f"  Time:             {elapsed/60:.1f} minutes")
print(f"  Speed:            {actual_count/elapsed:.0f} img/s")
print(f"  Output:           {OUTPUT_DIR}")
print(f"{'='*60}")
print(f"\n📋 Next steps:")
print(f"   1. Save this output as a Kaggle dataset")
print(f"   2. In training script, add as DATASET5_PATH:")
print(f"      Config.DATASET5_PATH = '/kaggle/input/your-synthetic-dataset'")
print(f"   3. Add it to build_image_list() scan")
