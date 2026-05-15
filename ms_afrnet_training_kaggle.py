"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                MS-AFR-Net: Multi-Scale AFR-Net Training Notebook            ║
║                                                                            ║
║  Complete training pipeline for:                                           ║
║    1. AFR-Net (Baseline)                                                   ║
║    2. MS-AFR-Net (Our Novel Model)                                         ║
║                                                                            ║
║  Configured for OFFLINE Kaggle (RTX 6000 Pro, internet OFF).              ║
║  Supports checkpoint-based multi-phase training.                           ║
║                                                                            ║
║  Input Datasets (attach in Kaggle):                                        ║
║    Dataset 1: AFR-Training-Real       (SOCOFing, NIST, FVC)               ║
║    Dataset 2: AFR-Training-Synthetic  (AMSL SGR/P2P)                      ║
║    Dataset 3: AFR-Training-Misc       (LivDet, MUST, UNFIT, ISPFDv1)     ║
║    Dataset 4: L3-SF v2                (Validation)                        ║
║    Dataset 5: resnet50-imagenet       (Pretrained weights - OFFLINE)      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import math
import random
import csv
import json
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from PIL import Image, ImageFilter
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
import torchvision.models as models

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           1. CONFIGURATION                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Config:
    # ── Paths ──
    DATASET1_PATH = "/kaggle/input/datasets/kushp3690/afr-training-real"
    DATASET2_PATH = "/kaggle/input/datasets/kushpatel7391/afr-training-synthetic"
    DATASET3_PATH = "/kaggle/input/datasets/kushpatel7391/afr-training-misc"
    DATASET4_PATH = "/kaggle/input/datasets/abvwmc/level-three-synthetic-fingerprint-generation-l3-sf"
    DATASET5_PATH = ""  # Generated synthetic fingerprints (set after running generate_synthetic_fingerprints.py)
    
    # ── Pretrained ResNet50 weights (for OFFLINE mode) ──
    # Upload resnet50_imagenet.pth as a Kaggle dataset and set path here
    RESNET50_WEIGHTS = "/kaggle/input/models/talakshrabari/resnet50-imagenet/pytorch/default/1"
    
    OUTPUT_DIR = "/kaggle/working"
    CHECKPOINT_DIR = "/kaggle/working/checkpoints"
    LOG_DIR = "/kaggle/working/logs"
    
    # ── Resume from previous session ──
    # After a Kaggle session ends, download the checkpoint and re-upload it
    # as a Kaggle dataset. Set the path below to point to it.
    # Example: "/kaggle/input/afr-checkpoints/afrnet_latest.pth"
    # Set to None if starting fresh or checkpoint is already in CHECKPOINT_DIR
    RESUME_FROM_INPUT = "/kaggle/input/models/kushpatel7391/afrnet-latest/pytorch/default/2"
    
    # ── Model Selection ──
    # Set to 'afrnet' for baseline, 'ms_afrnet' for our model
    MODEL_NAME = "afrnet"  # Change to "ms_afrnet" for Phase 2
    
    # ── Architecture (from Table 1 of the paper) ──
    INPUT_SIZE = 224
    INPUT_CHANNELS = 3
    EMBEDDING_DIM = 384
    
    # ── STN ──
    STN_ENABLED = True
    
    # ── Attention Head (AFR-Net) ──
    ATTN_NUM_HEADS = 6
    ATTN_NUM_LAYERS = 12        # 12 Transformer encoder layers
    ATTN_DROPOUT = 0.1
    
    # ── MS-AFR-Net specific ──
    MS_NUM_LAYERS = 6           # 6 cross-scale Transformer blocks
    MS_NUM_HEADS = 6
    MS_DROPOUT = 0.1
    
    # ── Training (Section 3.5 of the paper — EXACT match) ──
    BATCH_SIZE = 64             # Paper: 64 for AFR-Net
    GRAD_ACCUMULATION = 1       # Paper: no accumulation (4 GPUs)
    LEARNING_RATE = 1e-4        # Paper: 1e-4
    WEIGHT_DECAY = 2e-5         # Paper: 2e-5
    LR_SCHEDULER = "polynomial" # Paper: polynomial, power=3, min=1e-5
    LR_POWER = 3
    LR_MIN = 1e-5
    MAX_EPOCHS = 75             # Paper: 75
    WARMUP_EPOCHS = 3
    
    # ── ArcFace (Section 3.5) ──
    ARCFACE_MARGIN = 0.5        # Paper: m=0.5
    ARCFACE_SCALE = 64          # Paper: s=64
    ARCFACE_K = 1               # Sub-centers: 1=standard, 3=SubCenter ArcFace
    
    # ── Memory Management ──
    USE_AMP = True              # Mixed precision (fp16)
    NUM_WORKERS = 4             # RTX 6000 Pro has more CPU cores
    PIN_MEMORY = True
    GRADIENT_CHECKPOINTING = True
    FREEZE_BACKBONE_STAGES = 2  # Freeze Conv1 + Conv2 of ResNet50
    
    # ── Checkpointing & Logging ──
    SAVE_EVERY_N_EPOCHS = 5
    VAL_EVERY_N_EPOCHS = 1
    EARLY_STOPPING_PATIENCE = 15
    
    # ── ImageNet normalization ──
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # ── Class Filtering ──
    MIN_IMAGES_PER_CLASS = 5  # Drop classes with fewer images (helps ArcFace)
    
    # ── Reproducibility ──
    SEED = 42
    
    # ── Debug mode (small subset) ──
    DEBUG = False
    DEBUG_SAMPLES = 2000

    @classmethod
    def print_config(cls):
        print("\n" + "="*60)
        print("  TRAINING CONFIGURATION")
        print("="*60)
        for key, val in sorted(vars(cls).items()):
            if not key.startswith('_') and key.isupper():
                print(f"  {key:30s} = {val}")
        print("="*60 + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.SEED)
Config.print_config()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       2. PREPROCESSING & DATASET                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Folders to EXCLUDE (not fingerprint images)
EXCLUDE_PATTERNS = [
    'segmentation masks', 'segmentation_masks', 'segmasks',
    'minutuae_groundtruth', 'minutiae_groundtruth', 'xyt',
    'annotations',
    'pore ground truth', 'ground truth',
    'doc/', 'src/', 'sfinge/',
    'readme', 'thumbs.db',
    'extended_gallery',       # MUST gallery overlaps with MOLF (testing)
]

# FVC subsets reserved for TESTING — must NOT appear in training
# FVC 2002 DB1A, DB2A, DB3A and FVC 2004 DB1A, DB2A, DB3A → Testing
# FVC 2000 ALL + FVC 2002/2004 DB4+B sets → Training
FVC_TEST_PATTERNS = [
    'fvc2002/dbs/db1a', 'fvc2002/dbs/db1_a', 'fvc2002/dbs/db1 a',
    'fvc2002/dbs/db2a', 'fvc2002/dbs/db2_a', 'fvc2002/dbs/db2 a',
    'fvc2002/dbs/db3a', 'fvc2002/dbs/db3_a', 'fvc2002/dbs/db3 a',
    'fvc2004/dbs/db1a', 'fvc2004/dbs/db1_a', 'fvc2004/dbs/db1 a',
    'fvc2004/dbs/db2a', 'fvc2004/dbs/db2_a', 'fvc2004/dbs/db2 a',
    'fvc2004/dbs/db3a', 'fvc2004/dbs/db3_a', 'fvc2004/dbs/db3 a',
]

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def should_exclude(filepath):
    """Check if a file path matches any exclusion pattern."""
    fp_lower = str(filepath).lower().replace('\\', '/')
    for pattern in EXCLUDE_PATTERNS:
        if pattern in fp_lower:
            return True
    # Exclude FVC test sets from training
    for pattern in FVC_TEST_PATTERNS:
        if pattern in fp_lower:
            return True
    return False


def safe_open_image(filepath):
    """Open and convert an image safely, handling all modes."""
    try:
        img = Image.open(filepath)
        
        # Filter corrupted/tiny images
        if img.width < 32 or img.height < 32:
            return None
        
        # Convert all modes to RGB
        mode = img.mode
        if mode == 'I;16' or mode == 'I':
            # 16-bit → 8-bit
            arr = np.array(img, dtype=np.float32)
            if arr.max() > 0:
                arr = (arr / arr.max() * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            img = Image.fromarray(arr, mode='L')
            img = img.convert('RGB')
        elif mode == 'RGBA':
            # Drop alpha channel
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif mode == 'P':
            img = img.convert('RGB')
        elif mode == '1':
            img = img.convert('L').convert('RGB')
        elif mode == 'L' or mode == 'LA':
            img = img.convert('RGB')
        elif mode != 'RGB':
            img = img.convert('RGB')
        
        return img
    except Exception:
        return None


def extract_subject_id(filepath, base_path):
    """Extract a subject identifier from the file path for subject-disjoint splitting."""
    rel = os.path.relpath(filepath, base_path).replace('\\', '/').lower()
    parts = Path(rel).parts
    fname = Path(filepath).stem.lower()
    
    top = parts[0] if len(parts) > 0 else ''
    
    # ── SOCOFing: filename like "100__M_Left_little_finger_CR.BMP"
    if 'sokoto' in rel or 'socofing' in rel:
        subj = fname.split('__')[0] if '__' in fname else fname.split('_')[0]
        return f"socofing_{subj}"
    
    # ── NIST SD: folder structure "sd302a/images/.../SUBJECTID_*"
    if top.startswith('sd30'):
        # NIST naming: typically subject ID is first part of filename
        subj = fname.split('_')[0] if '_' in fname else fname[:5]
        return f"{top}_{subj}"
    
    # ── FVC: filename like "101_1.tif" → subject = 101
    if '74034' in rel or 'fvc' in rel:
        subj = fname.split('_')[0]
        # Identify which FVC subset
        for year in ['2000', '2002', '2004']:
            if year in rel:
                for db in ['db1', 'db2', 'db3', 'db4']:
                    if db in rel:
                        return f"fvc{year}_{db}_{subj}"
        return f"fvc_{subj}"
    
    # ── AMSL Synthetic: organized as subject001/..., subject002/...
    if 'amsl' in rel:
        for p in parts:
            if p.startswith('subject'):
                return f"amsl_{p}"
        # SGR v1 has no subject folders — each image is independent
        if 'sgr' in rel:
            return f"amsl_sgr_{fname}"
        return f"amsl_{fname}"
    
    # ── LivDet: use filename as pseudo-subject
    if 'livdet' in rel:
        subj = fname.split('_')[0] if '_' in fname else fname[:4]
        return f"livdet_{subj}"
    
    # ── MUST: organized by subject folders (10378, 11387, ...)
    if 'must' in rel:
        for p in parts:
            if p.isdigit() and len(p) >= 4:
                return f"must_{p}"
        return f"must_{fname[:5]}"
    
    # ── UNFIT
    if 'unfit' in rel:
        subj = fname.split('_')[0] if '_' in fname else fname[:4]
        return f"unfit_{subj}"
    
    # ── ISPFDv1: filename like "10_i_1_n_3.jpg" → subject = 10
    if 'ispfd' in rel:
        subj = fname.split('_')[0]
        return f"ispfd_{subj}"
    
    # ── L3-SF v2
    if 'l3sf' in rel or 'l3-sf' in rel:
        subj = fname.split('_')[0] if '_' in fname else fname[:4]
        return f"l3sf_{subj}"
    
    # Fallback
    return f"unknown_{fname[:8]}"


def scan_dataset(base_path, dataset_name="dataset"):
    """Scan a dataset directory and return list of valid image paths."""
    if not os.path.exists(base_path):
        print(f"  ⚠️  Path not found: {base_path}")
        return []
    
    all_images = []
    skipped = 0
    
    for root, dirs, files in os.walk(base_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            
            fpath = os.path.join(root, f)
            
            if should_exclude(fpath):
                skipped += 1
                continue
            
            all_images.append(fpath)
    
    print(f"  📂 {dataset_name}: {len(all_images):,} images (skipped {skipped:,} excluded)")
    return all_images


def get_sd302_subject_split(seed=42):
    """
    Split NIST SD302 subjects into 80/10/10 (train/val/test).
    Returns sets of subject IDs for each split.
    Subject IDs are 8-digit strings like '00002302'.
    """
    # Known SD302 subject range (from dataset analysis)
    # Extract all unique subjects from filenames: first 8 chars
    # We'll build this dynamically when scanning
    return seed  # Placeholder — actual split done in build_image_list


def build_image_list():
    """Build the master list of all training and validation images.
    
    Implements:
    - FVC 2002/2004 DB1A-3A exclusion (reserved for testing)
    - MUST Extended_gallery exclusion (overlaps MOLF testing)
    - NIST SD302 subject-level 80/10/10 split
    """
    print("\n🔍 Scanning all datasets...")
    print("  🔒 Excluding FVC 2002/2004 DB1A-3A (reserved for testing)")
    print("  🔒 Excluding MUST Extended_gallery (overlaps MOLF testing)")
    
    train_images = []
    val_images = []
    
    # Dataset 1-3 → Training (with exclusions handled by should_exclude)
    for path, name in [
        (Config.DATASET1_PATH, "Dataset1-Real"),
        (Config.DATASET2_PATH, "Dataset2-Synthetic"),
        (Config.DATASET3_PATH, "Dataset3-Misc"),
    ]:
        imgs = scan_dataset(path, name)
        train_images.extend([(p, path) for p in imgs])
    
    # ── NIST SD302 subject-level split (80% train / 10% val / 10% test) ──
    # Collect all SD302 subjects from training set
    sd302_subjects = set()
    sd302_images = []  # (filepath, base_path)
    non_sd302_images = []
    
    for fpath, base_path in train_images:
        rel = os.path.relpath(fpath, base_path).replace('\\', '/').lower()
        if rel.startswith('sd302') or rel.startswith('sd302a') or rel.startswith('sd302b') or rel.startswith('sd302d') or rel.startswith('sd302e'):
            # Extract subject ID (first 8 chars of filename)
            fname = os.path.basename(fpath)
            subj = fname.split('_')[0] if '_' in fname else fname[:8]
            sd302_subjects.add(subj)
            sd302_images.append((fpath, base_path, subj))
        else:
            non_sd302_images.append((fpath, base_path))
    
    if sd302_subjects:
        # Deterministic split
        sorted_subjects = sorted(sd302_subjects)
        rng = random.Random(Config.SEED)
        rng.shuffle(sorted_subjects)
        
        n = len(sorted_subjects)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        
        train_subjects = set(sorted_subjects[:n_train])
        val_subjects = set(sorted_subjects[n_train:n_train + n_val])
        test_subjects = set(sorted_subjects[n_train + n_val:])
        
        sd302_train = [(fp, bp) for fp, bp, s in sd302_images if s in train_subjects]
        sd302_val = [(fp, bp) for fp, bp, s in sd302_images if s in val_subjects]
        sd302_test_count = sum(1 for _, _, s in sd302_images if s in test_subjects)
        
        print(f"  📋 SD302 split: {len(train_subjects)} train / {len(val_subjects)} val / {len(test_subjects)} test subjects")
        print(f"     Images: {len(sd302_train)} train / {len(sd302_val)} val / {sd302_test_count} test (excluded)")
        
        train_images = non_sd302_images + sd302_train
        val_images.extend(sd302_val)
    
    # Dataset 4 → Validation (L3-SF v2)
    d4_imgs = scan_dataset(Config.DATASET4_PATH, "Dataset4-Validation")
    d4_filtered = [p for p in d4_imgs if 'ground truth' not in p.lower() and 'pore' not in p.lower()]
    val_images.extend([(p, Config.DATASET4_PATH) for p in d4_filtered])
    
    # Dataset 5 → Extra Synthetic (generated from StyleGAN2)
    if getattr(Config, 'DATASET5_PATH', '') and os.path.exists(Config.DATASET5_PATH):
        d5_imgs = scan_dataset(Config.DATASET5_PATH, "Dataset5-Generated")
        train_images.extend([(p, Config.DATASET5_PATH) for p in d5_imgs])
    
    print(f"\n✅ Total training images: {len(train_images):,}")
    print(f"✅ Total validation images: {len(val_images):,}")
    
    # ── Fingerprint Type Breakdown ──
    print("\n" + "="*70)
    print("  TRAINING DATA — FINGERPRINT TYPE BREAKDOWN")
    print("="*70)
    print(f"  {'Source':<30} {'Type':<25} {'Sensor/Method'}")
    print("-"*70)
    print(f"  {'SOCOFing':<30} {'Live — Optical':<25} {'Hamster Plus HSDU03P'}")
    print(f"  {'NIST SD302a':<30} {'Live — Rolled/Plain':<25} {'Crossmatch (rolled)'}")
    print(f"  {'NIST SD302b':<30} {'Latent':<25} {'Crime-scene lifts'}")
    print(f"  {'NIST SD302d':<30} {'Live — Contactless':<25} {'Smartphone camera'}")
    print(f"  {'NIST SD302e':<30} {'Live — Plain':<25} {'Crossmatch (plain)'}")
    print(f"  {'FVC 2000 (DB1-4B)':<30} {'Live — Mixed':<25} {'Optical/Capacitive/Thermal'}")
    print(f"  {'FVC 2002 (DB4+B sets)':<30} {'Live — Mixed':<25} {'Optical/Capacitive/Synth'}")
    print(f"  {'FVC 2004 (DB4+B sets)':<30} {'Live — Mixed':<25} {'Optical/Capacitive/Thermal'}")
    print(f"  {'AMSL (SGR/P2P)':<30} {'Synthetic':<25} {'GAN-generated'}")
    print(f"  {'LivDet 2009-2021':<30} {'Live + Spoof':<25} {'Various sensors'}")
    print(f"  {'MUST':<30} {'Latent':<25} {'Latent prints'}")
    print(f"  {'UNFIT':<30} {'Low-quality Live':<25} {'Degraded fingerprints'}")
    print(f"  {'ISPFDv1':<30} {'Fingerphoto':<25} {'Smartphone camera'}")
    print(f"  {'L3-SF v2 (val only)':<30} {'Synthetic':<25} {'High-res synthetic'}")
    print("="*70)
    
    if Config.DEBUG:
        random.shuffle(train_images)
        train_images = train_images[:Config.DEBUG_SAMPLES]
        val_images = val_images[:min(500, len(val_images))]
        print(f"  🐛 DEBUG MODE: Using {len(train_images)} train, {len(val_images)} val")
    
    return train_images, val_images


def assign_numeric_labels(image_list):
    """Assign numeric class labels based on subject IDs.
    Filters out classes with fewer than MIN_IMAGES_PER_CLASS samples."""
    image_subjects = []
    for fpath, base_path in image_list:
        sid = extract_subject_id(fpath, base_path)
        image_subjects.append(sid)
    
    # Count images per class
    class_counts = defaultdict(int)
    for sid in image_subjects:
        class_counts[sid] += 1
    
    total_classes = len(class_counts)
    
    # Filter: keep only classes with enough samples
    min_imgs = getattr(Config, 'MIN_IMAGES_PER_CLASS', 1)
    valid_classes = {sid for sid, count in class_counts.items() if count >= min_imgs}
    
    # Rebuild filtered image list and labels
    filtered_images = []
    filtered_subjects = []
    for i, (fpath_bp, sid) in enumerate(zip(image_list, image_subjects)):
        if sid in valid_classes:
            filtered_images.append(fpath_bp)
            filtered_subjects.append(sid)
    
    dropped = total_classes - len(valid_classes)
    dropped_imgs = len(image_list) - len(filtered_images)
    
    sid_to_label = {sid: idx for idx, sid in enumerate(sorted(valid_classes))}
    labels = [sid_to_label[sid] for sid in filtered_subjects]
    
    print(f"  📋 Total subjects found: {total_classes:,}")
    print(f"  🗑️  Dropped {dropped:,} classes with < {min_imgs} images ({dropped_imgs:,} images removed)")
    print(f"  ✅ Keeping {len(valid_classes):,} classes, {len(filtered_images):,} images")
    return labels, sid_to_label, filtered_images


class FingerprintDataset(Dataset):
    """Unified fingerprint dataset for all data sources."""
    
    def __init__(self, image_list, labels, transform=None, is_training=True):
        self.image_list = image_list    # List of (filepath, base_path)
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        fpath, base_path = self.image_list[idx]
        label = self.labels[idx]
        
        img = safe_open_image(fpath)
        
        if img is None:
            # Return a black image on error
            img = Image.new('RGB', (Config.INPUT_SIZE, Config.INPUT_SIZE), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_transforms(is_training=True):
    """Get the preprocessing + augmentation transforms."""
    if is_training:
        return T.Compose([
            T.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomRotation(degrees=20, fill=255),
            T.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.90, 1.10), fill=255),
            T.RandomPerspective(distortion_scale=0.15, p=0.3, fill=255),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
            T.RandomGrayscale(p=0.15),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            T.ToTensor(),
            T.Normalize(mean=Config.MEAN, std=Config.STD),
            T.RandomErasing(p=0.35, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
        ])
    else:
        return T.Compose([
            T.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=Config.MEAN, std=Config.STD),
        ])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              2.5: PREPROCESSING VISUALIZATION                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def visualize_preprocessing(num_samples=3, save_dir="/kaggle/working"):
    """
    Visualize before/after preprocessing for sample images from each dataset.
    Shows: Original → Resized (224×224) → Augmented
    Saves the comparison grid to save_dir.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    os.makedirs(save_dir, exist_ok=True)
    
    datasets_info = [
        (Config.DATASET1_PATH, "Dataset1-Real\n(SOCOFing/NIST/FVC)"),
        (Config.DATASET2_PATH, "Dataset2-Synthetic\n(AMSL/P2P)"),
        (Config.DATASET3_PATH, "Dataset3-Misc\n(LivDet/MUST/UNFIT)"),
        (Config.DATASET4_PATH, "Dataset4-Validation\n(L3-SF v2)"),
    ]
    
    # Transforms
    resize_only = T.Compose([
        T.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    ])
    to_tensor_norm = T.Compose([
        T.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=Config.MEAN, std=Config.STD),
    ])
    train_aug = get_transforms(is_training=True)
    
    # Inverse normalization for display
    inv_mean = torch.tensor(Config.MEAN).view(3, 1, 1)
    inv_std = torch.tensor(Config.STD).view(3, 1, 1)
    
    def tensor_to_pil(t):
        """Convert normalized tensor back to displayable image."""
        t = t.clone()
        t = t * inv_std + inv_mean
        t = t.clamp(0, 1)
        return T.ToPILImage()(t)
    
    print("\n🎨 Generating Preprocessing Visualization...")
    
    # ── 1. Collect sample images from each dataset ──
    all_samples = []  # List of (original_pil, dataset_name, filename, original_size, mode)
    
    for ds_path, ds_name in datasets_info:
        if not os.path.exists(ds_path):
            print(f"  ⚠️  Skipping {ds_name}: path not found")
            continue
        
        # Walk and collect valid images
        candidates = []
        for root, dirs, files in os.walk(ds_path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTENSIONS and not should_exclude(os.path.join(root, f)):
                    candidates.append(os.path.join(root, f))
                    if len(candidates) >= 500:  # Don't scan forever
                        break
            if len(candidates) >= 500:
                break
        
        # Random sample
        random.shuffle(candidates)
        for fpath in candidates[:num_samples]:
            try:
                raw_img = Image.open(fpath)
                original_size = f"{raw_img.width}×{raw_img.height}"
                original_mode = raw_img.mode
                
                # Keep a copy before conversion
                if raw_img.mode in ('L', 'I;16', 'I'):
                    display_original = raw_img.convert('RGB')
                elif raw_img.mode == 'RGBA':
                    bg = Image.new('RGB', raw_img.size, (255, 255, 255))
                    bg.paste(raw_img, mask=raw_img.split()[3])
                    display_original = bg
                else:
                    display_original = raw_img.convert('RGB')
                
                # Processed version
                processed = safe_open_image(fpath)
                if processed is None:
                    continue
                
                fname = os.path.basename(fpath)
                all_samples.append((display_original, processed, ds_name, fname, original_size, original_mode))
            except Exception:
                continue
    
    if not all_samples:
        print("  ⚠️  No samples found!")
        return
    
    # ── 2. Create the big comparison figure ──
    n_rows = len(all_samples)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('MS-AFR-Net Preprocessing Pipeline Visualization', 
                 fontsize=20, fontweight='bold', y=1.02)
    
    col_titles = ['Original Image', 'After RGB Conversion', 'Resized (224×224)', 'With Augmentation']
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    for i, (display_orig, processed, ds_name, fname, orig_size, orig_mode) in enumerate(all_samples):
        # Col 0: Original (as-is, with original resolution)
        axes[i, 0].imshow(display_orig)
        axes[i, 0].set_ylabel(f"{ds_name}\n{fname}\n{orig_size} | {orig_mode}", 
                              fontsize=9, rotation=0, labelpad=120, ha='right', va='center')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        
        # Col 1: After RGB conversion (safe_open_image output)
        axes[i, 1].imshow(processed)
        axes[i, 1].set_xlabel(f"{processed.width}×{processed.height} RGB", fontsize=9)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        
        # Col 2: After resize to 224×224 (no augmentation)
        resized = resize_only(processed)
        axes[i, 2].imshow(resized)
        axes[i, 2].set_xlabel("224×224 RGB", fontsize=9)
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        
        # Col 3: With training augmentation (random rotation, jitter, etc.)
        aug_tensor = train_aug(processed)
        aug_pil = tensor_to_pil(aug_tensor)
        axes[i, 3].imshow(aug_pil)
        axes[i, 3].set_xlabel("224×224 + Augmented", fontsize=9)
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])
        
        # Color borders
        for j in range(4):
            for spine in axes[i, j].spines.values():
                spine.set_edgecolor(['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][j])
                spine.set_linewidth(2)
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(save_dir, "preprocessing_visualization.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  💾 Saved: {save_path}")
    
    # ── 3. Also create a smaller summary of image mode distributions ──
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Dataset Image Properties', fontsize=16, fontweight='bold')
    
    # Sample more images for statistics
    modes = []
    sizes = []
    for ds_path, ds_name in datasets_info:
        if not os.path.exists(ds_path):
            continue
        count = 0
        for root, dirs, files in os.walk(ds_path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTENSIONS and not should_exclude(os.path.join(root, f)):
                    try:
                        with Image.open(os.path.join(root, f)) as img:
                            modes.append(img.mode)
                            sizes.append(max(img.width, img.height))
                            count += 1
                    except Exception:
                        pass
                    if count >= 200:
                        break
            if count >= 200:
                break
    
    # Mode distribution
    mode_counts = Counter(modes)
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    axes2[0].bar(mode_counts.keys(), mode_counts.values(), color=colors[:len(mode_counts)])
    axes2[0].set_title('Image Mode Distribution (sample)', fontsize=12)
    axes2[0].set_ylabel('Count')
    
    # Size distribution
    axes2[1].hist(sizes, bins=30, color='#3498db', edgecolor='white', alpha=0.8)
    axes2[1].axvline(x=224, color='#e74c3c', linestyle='--', linewidth=2, label='Target: 224px')
    axes2[1].set_title('Max Dimension Distribution (sample)', fontsize=12)
    axes2[1].set_xlabel('Pixels')
    axes2[1].set_ylabel('Count')
    axes2[1].legend()
    
    plt.tight_layout()
    stats_path = os.path.join(save_dir, "dataset_statistics.png")
    fig2.savefig(stats_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"  💾 Saved: {stats_path}")
    
    # ── 4. Save individual augmentation steps per image ──
    aug_dir = os.path.join(save_dir, "augmentation_samples")
    os.makedirs(aug_dir, exist_ok=True)
    
    # Individual augmentation transforms
    aug_steps = {
        "1_original": None,
        "2_rgb_converted": None,  # safe_open_image output
        "3_resized_224": T.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE), 
                                   interpolation=T.InterpolationMode.BILINEAR),
        "4_rotated": T.Compose([
            T.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            T.RandomRotation(degrees=15),
        ]),
        "5_affine": T.Compose([
            T.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]),
        "6_hflip": T.Compose([
            T.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            T.RandomHorizontalFlip(p=1.0),  # Force flip so we can see it
        ]),
        "7_color_jitter": T.Compose([
            T.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ]),
        "8_grayscale": T.Compose([
            T.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            T.RandomGrayscale(p=1.0),  # Force grayscale
        ]),
        "9_all_augmented": get_transforms(is_training=True),
    }
    
    for i, (display_orig, processed, ds_name, fname, orig_size, orig_mode) in enumerate(all_samples):
        # Create per-image directory
        clean_dsname = ds_name.replace("\n", "_").replace("/", "-").replace("(", "").replace(")", "")
        img_dir = os.path.join(aug_dir, f"{i+1:02d}_{clean_dsname}_{fname.split('.')[0]}")
        os.makedirs(img_dir, exist_ok=True)
        
        # Save original
        display_orig.save(os.path.join(img_dir, "1_original.png"))
        
        # Save RGB converted
        processed.save(os.path.join(img_dir, "2_rgb_converted.png"))
        
        # Save each augmentation step
        for step_name, transform in aug_steps.items():
            if step_name in ("1_original", "2_rgb_converted"):
                continue  # Already saved
            
            try:
                if step_name == "9_all_augmented":
                    # This returns tensor, need to convert back
                    tensor_out = transform(processed)
                    out_img = tensor_to_pil(tensor_out)
                    # Also save 3 different random versions to show variety
                    out_img.save(os.path.join(img_dir, f"{step_name}_v1.png"))
                    for v in range(2, 4):
                        tensor_v = transform(processed)
                        pil_v = tensor_to_pil(tensor_v)
                        pil_v.save(os.path.join(img_dir, f"{step_name}_v{v}.png"))
                else:
                    out_img = transform(processed)
                    out_img.save(os.path.join(img_dir, f"{step_name}.png"))
            except Exception as e:
                print(f"    ⚠️ {step_name} failed for {fname}: {e}")
    
    print(f"  💾 Individual augmentation steps saved to: {aug_dir}")
    print(f"     {len(all_samples)} images × 10 augmentation steps")
    
    print("  ✅ Visualization complete!\n")


def visualize_fingerprint_types(save_dir="/kaggle/working"):
    """
    Display sample images from each fingerprint TYPE in the training data.
    Creates a labeled grid: rows = fingerprint types, cols = sample images.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Map fingerprint types → (search keywords in path, dataset path, color)
    TYPE_MAP = {
        'Live — Optical (SOCOFing)':    (['sokoto', 'socofing'],          Config.DATASET1_PATH, '#3498db'),
        'Live — Rolled (NIST SD302a)':  (['sd302a'],                      Config.DATASET1_PATH, '#2980b9'),
        'Latent (NIST SD302b)':         (['sd302b'],                      Config.DATASET1_PATH, '#e74c3c'),
        'Contactless (NIST SD302d)':    (['sd302d'],                      Config.DATASET1_PATH, '#9b59b6'),
        'Live — FVC (Optical/Cap)':     (['fvc'],                         Config.DATASET1_PATH, '#2ecc71'),
        'Synthetic (AMSL)':             (['amsl'],                        Config.DATASET2_PATH, '#f39c12'),
        'Live + Spoof (LivDet)':        (['livdet'],                      Config.DATASET3_PATH, '#e67e22'),
        'Latent (MUST)':                (['must'],                        Config.DATASET3_PATH, '#c0392b'),
        'Low-Quality (UNFIT)':          (['unfit'],                       Config.DATASET3_PATH, '#7f8c8d'),
        'Fingerphoto (ISPFDv1)':        (['ispfd'],                       Config.DATASET3_PATH, '#1abc9c'),
        'Synthetic (L3-SF v2)':         (['l3sf', 'l3-sf'],              Config.DATASET4_PATH, '#f1c40f'),
    }
    
    print("\n🎨 Generating Fingerprint Type Gallery...")
    
    n_samples = 4  # images per type
    collected = {}  # type_name -> list of PIL images
    
    for type_name, (keywords, base_path, color) in TYPE_MAP.items():
        if not os.path.exists(base_path):
            continue
        
        candidates = []
        for root, dirs, files in os.walk(base_path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    continue
                fpath = os.path.join(root, f)
                if should_exclude(fpath):
                    continue
                fp_low = fpath.lower().replace('\\', '/')
                if any(kw in fp_low for kw in keywords):
                    candidates.append(fpath)
                if len(candidates) >= 200:
                    break
            if len(candidates) >= 200:
                break
        
        if not candidates:
            continue
        
        random.shuffle(candidates)
        imgs = []
        for fp in candidates[:n_samples * 3]:
            img = safe_open_image(fp)
            if img is not None:
                img = img.resize((224, 224), Image.BILINEAR)
                imgs.append(img)
            if len(imgs) >= n_samples:
                break
        
        if imgs:
            collected[type_name] = imgs
            print(f"  ✅ {type_name}: found {len(candidates)} images, showing {len(imgs)}")
    
    if not collected:
        print("  ⚠️ No fingerprint types found, skipping gallery")
        return
    
    # Create grid: rows = types, cols = samples
    n_types = len(collected)
    fig, axes = plt.subplots(n_types, n_samples, figsize=(n_samples * 3.5, n_types * 3.5))
    fig.suptitle('Training Data — Fingerprint Type Gallery', fontsize=18, fontweight='bold', y=1.01)
    
    if n_types == 1:
        axes = axes.reshape(1, -1)
    
    type_names = list(collected.keys())
    for i, type_name in enumerate(type_names):
        imgs = collected[type_name]
        color = TYPE_MAP[type_name][2]
        
        for j in range(n_samples):
            ax = axes[i, j]
            if j < len(imgs):
                ax.imshow(imgs[j])
            else:
                ax.set_facecolor('#f0f0f0')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
        
        # Row label
        axes[i, 0].set_ylabel(type_name, fontsize=10, fontweight='bold',
                               rotation=0, labelpad=140, ha='right', va='center')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "fingerprint_type_gallery.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  💾 Saved: {save_path}")
    print("  ✅ Fingerprint type gallery complete!\n")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    3. MODEL ARCHITECTURE                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 3.1: Spatial Transformer Network (STN)                                     │
# │ Reference: Jaderberg et al., "Spatial Transformer Networks" (2015)          │
# │ EXACT match to Table 1 of AFR-Net paper (Grosz & Jain, 2024)               │
# └──────────────────────────────────────────────────────────────────────────────┘

class SpatialTransformerNetwork(nn.Module):
    """
    Spatial Alignment Module — EXACT paper architecture (Table 1).
    
    5 Conv+MaxPool blocks (channels: 16→24→32→48→64) → Flatten
    → FC(3136→32) → FC(32→4) → (s, θ, tx, ty)
    → Construct 2×3 affine matrix → Grid sample
    
    Outputs 4 interpretable params instead of 6 raw affine values.
    """
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Paper Table 1: Loc1-Loc10
        self.localization = nn.Sequential(
            # Loc1: Conv2d 16×224×224, k=7x7, padding=3
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            # Loc2: MaxPool 16×112×112, k=2x2, stride=2
            nn.MaxPool2d(2, stride=2),
            
            # Loc3: Conv2d 24×112×112, k=5x5, padding=2
            nn.Conv2d(16, 24, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            # Loc4: MaxPool 24×56×56
            nn.MaxPool2d(2, stride=2),
            
            # Loc5: Conv2d 32×56×56, k=3x3, padding=1
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Loc6: MaxPool 32×28×28
            nn.MaxPool2d(2, stride=2),
            
            # Loc7: Conv2d 48×28×28, k=3x3, padding=1
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Loc8: MaxPool 48×14×14
            nn.MaxPool2d(2, stride=2),
            
            # Loc9: Conv2d 64×14×14, k=3x3, padding=1
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Loc10: MaxPool 64×7×7
            nn.MaxPool2d(2, stride=2),
        )
        
        # Loc11: Linear(64*7*7=3136 → 32) + Loc12: Linear(32 → 4)
        # Outputs: (scale, rotation_angle, tx, ty)
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4),
        )
        
        # Initialize to identity: s=1, θ=0, tx=0, ty=0
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float))
    
    def forward(self, x):
        B = x.size(0)
        
        # Localization network → 4 params
        features = self.localization(x)
        features = features.view(B, -1)  # [B, 3136]
        params = self.fc(features)        # [B, 4] = (s, θ, tx, ty)
        
        s     = params[:, 0]  # scale
        theta = params[:, 1]  # rotation angle (radians)
        tx    = params[:, 2]  # horizontal translation
        ty    = params[:, 3]  # vertical translation
        
        # Construct 2×3 affine matrix from (s, θ, tx, ty)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        # [s·cos(θ)  -s·sin(θ)  tx]
        # [s·sin(θ)   s·cos(θ)  ty]
        affine = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
        affine[:, 0, 0] = s * cos_t
        affine[:, 0, 1] = -s * sin_t
        affine[:, 0, 2] = tx
        affine[:, 1, 0] = s * sin_t
        affine[:, 1, 1] = s * cos_t
        affine[:, 1, 2] = ty
        
        grid = F.affine_grid(affine, x.size(), align_corners=False)
        x_transformed = F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='border')
        
        return x_transformed


# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 3.2: ResNet50 Backbone (Feature Extractor)                                │
# └──────────────────────────────────────────────────────────────────────────────┘

class ResNet50Backbone(nn.Module):
    """
    ResNet50 backbone pretrained on ImageNet.
    
    Returns intermediate feature maps for multi-scale processing:
        conv3_x: 28×28×512
        conv4_x: 14×14×1024
        conv5_x:  7×7×2048
    """
    
    def __init__(self, pretrained=True, freeze_stages=2, gradient_checkpointing=False):
        super().__init__()
        
        # Load ResNet50 — offline-safe: use local weights file
        weights_path = Config.RESNET50_WEIGHTS
        # If path is a directory, append the known filename
        if os.path.isdir(weights_path):
            weights_path = os.path.join(weights_path, "resnet50_imagenet.pth")
        
        if pretrained and os.path.isfile(weights_path):
            print(f"  📦 Loading ResNet50 weights from: {weights_path}")
            resnet = models.resnet50(weights=None)
            state = torch.load(weights_path, map_location='cpu')
            resnet.load_state_dict(state)
        elif pretrained:
            # Fallback — should NOT happen with internet off
            print(f"  ⚠️ Local weights not found at: {weights_path}")
            print(f"  🌐 Attempting download (will fail without internet)...")
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet50(weights=None)
        
        # Split ResNet50 into stages
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # 56×56
        self.layer1 = resnet.layer1   # conv2_x: 56×56×256
        self.layer2 = resnet.layer2   # conv3_x: 28×28×512
        self.layer3 = resnet.layer3   # conv4_x: 14×14×1024
        self.layer4 = resnet.layer4   # conv5_x:  7×7×2048
        
        self.gradient_checkpointing = gradient_checkpointing
        
        # Freeze early stages
        if freeze_stages >= 1:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False
        if freeze_stages >= 2:
            for param in self.layer2.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        
        if self.gradient_checkpointing and self.training:
            conv3 = torch.utils.checkpoint.checkpoint(self.layer2, x, use_reentrant=False)
            conv4 = torch.utils.checkpoint.checkpoint(self.layer3, conv3, use_reentrant=False)
            conv5 = torch.utils.checkpoint.checkpoint(self.layer4, conv4, use_reentrant=False)
        else:
            conv3 = self.layer2(x)     # 28×28×512
            conv4 = self.layer3(conv3)  # 14×14×1024
            conv5 = self.layer4(conv4)  #  7×7×2048
        
        return conv3, conv4, conv5


# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 3.3: CNN Classification Head (Z_c)                                        │
# │ Produces a 384-D embedding from conv5_x features                          │
# └──────────────────────────────────────────────────────────────────────────────┘

class CNNHead(nn.Module):
    """
    CNN Head: conv5_x → GAP → FC → Z_c (384-D)
    
    This is the standard CNN pathway that captures global features.
    """
    
    def __init__(self, in_channels=2048, embed_dim=384):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Paper Table 1, Row 18: Zc = Linear(2048→384, bias=True)
        self.fc = nn.Linear(in_channels, embed_dim, bias=True)
    
    def forward(self, conv5):
        x = self.gap(conv5)          # [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)   # [B, 2048]
        z_c = self.fc(x)             # [B, 384]
        return z_c


# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 3.4: Attention Head (Z_a) — Original AFR-Net                              │
# │ Uses conv4_x features with 12 Transformer encoder layers                  │
# └──────────────────────────────────────────────────────────────────────────────┘

class AttentionHead(nn.Module):
    """
    Attention Head (Original AFR-Net):
        conv4_x (14×14×1024) → Linear Projection → 196 tokens × 384-D
        → [CLS] token prepended → Positional Encoding
        → 12 Transformer Encoder layers (6 heads, 384-D)
        → CLS token output → Z_a (384-D)
    """
    
    def __init__(self, in_channels=1024, embed_dim=384, num_heads=6, num_layers=12, dropout=0.1):
        super().__init__()
        
        # Paper Table 1, Row 19: MLP projection (in=1024, hid=1024, out=384)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, embed_dim, kernel_size=1),
        )
        
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Positional encoding: 196 spatial tokens + 1 CLS = 197
        self.pos_embed = nn.Parameter(torch.randn(1, 197, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, conv4):
        B = conv4.size(0)
        
        # Project: [B, 1024, 14, 14] → [B, 384, 14, 14]
        x = self.proj(conv4)
        
        # Flatten spatial: [B, 384, 14, 14] → [B, 196, 384]
        x = x.flatten(2).transpose(1, 2)
        
        # Prepend CLS token: [B, 197, 384]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        # Add positional encoding
        x = self.pos_drop(x + self.pos_embed)
        
        # Transformer
        x = self.transformer(x)
        
        # Extract CLS token
        z_a = self.norm(x[:, 0])  # [B, 384]
        
        return z_a


# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 3.5: Multi-Scale Feature Pyramid (NEW — MS-AFR-Net)                       │
# │ Extracts and projects features from conv3, conv4, conv5                   │
# └──────────────────────────────────────────────────────────────────────────────┘

class MultiScaleFeaturePyramid(nn.Module):
    """
    Multi-Scale Feature Pyramid (Our Contribution):
    
    Takes features from three ResNet stages and projects them to a common 
    embedding dimension:
        conv3_x (28×28×512)  → Linear Proj → 784 tokens × 384-D  (fine detail)
        conv4_x (14×14×1024) → Linear Proj → 196 tokens × 384-D  (medium)
        conv5_x (7×7×2048)   → Linear Proj →  49 tokens × 384-D  (coarse/global)
    
    Total: 1,029 tokens × 384-D → fed to Cross-Scale Attention
    """
    
    def __init__(self, embed_dim=384):
        super().__init__()
        
        # Projections for each scale
        self.proj3 = nn.Sequential(
            nn.Conv2d(512, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        
        self.proj4 = nn.Sequential(
            nn.Conv2d(1024, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        
        self.proj5 = nn.Sequential(
            nn.Conv2d(2048, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        
        # Scale-specific learnable embeddings (to distinguish scales)
        self.scale_embed_3 = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.scale_embed_4 = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.scale_embed_5 = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Positional encodings per scale
        self.pos_3 = nn.Parameter(torch.randn(1, 784, embed_dim) * 0.02)  # 28×28
        self.pos_4 = nn.Parameter(torch.randn(1, 196, embed_dim) * 0.02)  # 14×14
        self.pos_5 = nn.Parameter(torch.randn(1, 49, embed_dim) * 0.02)   # 7×7
    
    def forward(self, conv3, conv4, conv5):
        B = conv3.size(0)
        
        # Project each scale
        f3 = self.proj3(conv3).flatten(2).transpose(1, 2)  # [B, 784, 384]
        f4 = self.proj4(conv4).flatten(2).transpose(1, 2)  # [B, 196, 384]
        f5 = self.proj5(conv5).flatten(2).transpose(1, 2)  # [B, 49, 384]
        
        # Add positional + scale embeddings
        f3 = f3 + self.pos_3 + self.scale_embed_3
        f4 = f4 + self.pos_4 + self.scale_embed_4
        f5 = f5 + self.pos_5 + self.scale_embed_5
        
        # Concatenate all scales: [B, 1029, 384]
        multi_scale_tokens = torch.cat([f3, f4, f5], dim=1)
        
        # Return individual counts for the fusion gate
        scale_lengths = (f3.size(1), f4.size(1), f5.size(1))
        
        return multi_scale_tokens, scale_lengths


# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 3.6: Cross-Scale Attention Module (NEW — MS-AFR-Net)                      │
# │ 6 Transformer blocks that process all 1,029 multi-scale tokens            │
# └──────────────────────────────────────────────────────────────────────────────┘

class CrossScaleAttention(nn.Module):
    """
    Cross-Scale Attention Module (Our Contribution):
    
    6 Transformer encoder layers that allow tokens from different scales
    (conv3, conv4, conv5) to attend to each other, learning cross-scale
    feature relationships.
    
    This is the key innovation: fine-scale minutiae tokens can attend to
    coarse-scale ridge flow tokens, creating a unified multi-scale representation.
    """
    
    def __init__(self, embed_dim=384, num_heads=6, num_layers=6, dropout=0.1):
        super().__init__()
        
        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, multi_scale_tokens):
        B = multi_scale_tokens.size(0)
        
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, multi_scale_tokens], dim=1)  # [B, 1030, 384]
        
        # Cross-scale Transformer
        x = self.transformer(x)
        
        # Output: CLS token captures the cross-scale representation
        cls_out = self.norm(x[:, 0])  # [B, 384]
        
        # Also return per-scale representations (for fusion gate)
        token_out = x[:, 1:]  # [B, 1029, 384]
        
        return cls_out, token_out


# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 3.7: Adaptive Scale Fusion Gate (NEW — MS-AFR-Net)                        │
# │ Learns optimal per-sample weights for combining multi-scale features      │
# └──────────────────────────────────────────────────────────────────────────────┘

class AdaptiveScaleFusionGate(nn.Module):
    """
    Adaptive Scale Fusion Gate (Our Contribution):
    
    Instead of treating all scales equally, this module learns a per-sample
    gating mechanism that weights the contribution of each scale based on the
    input image characteristics.
    
    For high-quality contact prints → coarse scale dominates
    For latent/partial prints → fine scale dominates
    
    Architecture:
        1. Pool each scale's tokens from Cross-Scale Attention output
        2. Concatenate with CLS token
        3. MLP → sigmoid gates for each scale
        4. Weighted sum → Z_ms (384-D)
    """
    
    def __init__(self, embed_dim=384):
        super().__init__()
        
        # Per-scale pooling: average pool each scale's tokens
        self.scale_proj = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),  # 4 = cls + 3 scales
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Gate network: outputs 3 gate values (one per scale)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 3),
            nn.Softmax(dim=-1),  # Ensures gates sum to 1
        )
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
    
    def forward(self, cls_out, token_out, scale_lengths):
        B = cls_out.size(0)
        n3, n4, n5 = scale_lengths
        
        # Split token_out back into per-scale groups
        tokens_3 = token_out[:, :n3]                      # [B, 784, 384]
        tokens_4 = token_out[:, n3:n3+n4]                  # [B, 196, 384]
        tokens_5 = token_out[:, n3+n4:n3+n4+n5]            # [B, 49, 384]
        
        # Pool each scale
        pool_3 = tokens_3.mean(dim=1)  # [B, 384]
        pool_4 = tokens_4.mean(dim=1)  # [B, 384]
        pool_5 = tokens_5.mean(dim=1)  # [B, 384]
        
        # Concatenate with CLS
        combined = torch.cat([cls_out, pool_3, pool_4, pool_5], dim=-1)  # [B, 1536]
        combined = self.scale_proj(combined)  # [B, 384]
        
        # Compute gate values
        gates = self.gate(combined)  # [B, 3]
        g3, g4, g5 = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
        
        # Weighted fusion
        z_ms = g3 * pool_3 + g4 * pool_4 + g5 * pool_5  # [B, 384]
        z_ms = self.output_proj(z_ms)  # [B, 384]
        
        return z_ms, gates


# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 3.8: Complete AFR-Net Model (Baseline)                                    │
# └──────────────────────────────────────────────────────────────────────────────┘

class AFRNet(nn.Module):
    """
    Original AFR-Net Architecture (Baseline).
    
    Input (3×224×224) → STN → ResNet50 backbone
      → CNN Head: Conv5_x → GAP → FC → Z_c (384-D)
      → Attention Head: Conv4_x → 12 Transformer blocks → Z_a (384-D)
    → Concatenate [Z_c; Z_a] = 768-D
    """
    
    def __init__(self, config=Config):
        super().__init__()
        
        self.stn = SpatialTransformerNetwork(in_channels=config.INPUT_CHANNELS) if config.STN_ENABLED else nn.Identity()
        
        self.backbone = ResNet50Backbone(
            pretrained=True,
            freeze_stages=config.FREEZE_BACKBONE_STAGES,
            gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        )
        
        self.cnn_head = CNNHead(
            in_channels=2048,
            embed_dim=config.EMBEDDING_DIM,
        )
        
        self.attention_head = AttentionHead(
            in_channels=1024,
            embed_dim=config.EMBEDDING_DIM,
            num_heads=config.ATTN_NUM_HEADS,
            num_layers=config.ATTN_NUM_LAYERS,
            dropout=config.ATTN_DROPOUT,
        )
        
        self.embed_dim = config.EMBEDDING_DIM * 2  # 768
    
    def forward(self, x):
        # STN alignment
        x = self.stn(x)
        
        # Backbone feature extraction
        conv3, conv4, conv5 = self.backbone(x)
        
        # Dual heads
        z_c = self.cnn_head(conv5)         # [B, 384]
        z_a = self.attention_head(conv4)   # [B, 384]
        
        # Concatenate embeddings
        embedding = torch.cat([z_c, z_a], dim=-1)  # [B, 768]
        
        return embedding


# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 3.9: Complete MS-AFR-Net Model (Our Novel Architecture)                   │
# └──────────────────────────────────────────────────────────────────────────────┘

class MSAFRNet(nn.Module):
    """
    MS-AFR-Net: Multi-Scale Attention-Driven Fingerprint Recognition Network.
    
    Our Novel Contribution:
    
    Input (3×224×224) → STN → ResNet50 backbone (SHARED)
      → CNN Head: Conv5_x → GAP → FC → Z_c (384-D)
      → Multi-Scale Feature Pyramid:
          Conv3_x (28×28×512)  → 784 tokens × 384-D
          Conv4_x (14×14×1024) → 196 tokens × 384-D
          Conv5_x (7×7×2048)   →  49 tokens × 384-D
      → Cross-Scale Attention: 6 Transformer blocks (1,029 tokens)
      → Adaptive Scale Fusion Gate → Z_ms (384-D)
    → Concatenate [Z_c; Z_ms] = 768-D
    """
    
    def __init__(self, config=Config):
        super().__init__()
        
        self.stn = SpatialTransformerNetwork(in_channels=config.INPUT_CHANNELS) if config.STN_ENABLED else nn.Identity()
        
        self.backbone = ResNet50Backbone(
            pretrained=True,
            freeze_stages=config.FREEZE_BACKBONE_STAGES,
            gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        )
        
        self.cnn_head = CNNHead(
            in_channels=2048,
            embed_dim=config.EMBEDDING_DIM,
        )
        
        self.feature_pyramid = MultiScaleFeaturePyramid(embed_dim=config.EMBEDDING_DIM)
        
        self.cross_attention = CrossScaleAttention(
            embed_dim=config.EMBEDDING_DIM,
            num_heads=config.MS_NUM_HEADS,
            num_layers=config.MS_NUM_LAYERS,
            dropout=config.MS_DROPOUT,
        )
        
        self.fusion_gate = AdaptiveScaleFusionGate(embed_dim=config.EMBEDDING_DIM)
        
        self.embed_dim = config.EMBEDDING_DIM * 2  # 768
    
    def forward(self, x):
        # STN alignment
        x = self.stn(x)
        
        # Backbone feature extraction
        conv3, conv4, conv5 = self.backbone(x)
        
        # CNN Head
        z_c = self.cnn_head(conv5)  # [B, 384]
        
        # Multi-Scale pathway
        ms_tokens, scale_lengths = self.feature_pyramid(conv3, conv4, conv5)
        cls_out, token_out = self.cross_attention(ms_tokens)
        z_ms, gates = self.fusion_gate(cls_out, token_out, scale_lengths)  # [B, 384]
        
        # Concatenate embeddings
        embedding = torch.cat([z_c, z_ms], dim=-1)  # [B, 768]
        
        return embedding


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        4. LOSS FUNCTION                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class ArcFaceLoss(nn.Module):
    """
    SubCenter ArcFace Loss (Deng et al., 2020).
    
    Standard ArcFace uses 1 center per class. With limited data (5-6 imgs/class),
    a single center can't capture intra-class variation (pose, pressure, sensor).
    
    SubCenter ArcFace uses K sub-centers per class:
    - Each class has K weight vectors instead of 1
    - Cosine similarity computed with all K sub-centers
    - Maximum similarity used as the class score
    - This lets different sub-centers handle different variations
    
    Set K=1 for standard ArcFace (paper baseline).
    Set K=3 for SubCenter ArcFace (recommended for limited data).
    
    Parameters (from AFR-Net Section 3.5):
        margin (m) = 0.5
        scale (s)  = 64
    """
    
    def __init__(self, embed_dim, num_classes, margin=0.5, scale=64, K=1):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.num_classes = num_classes
        self.K = K  # Number of sub-centers per class
        
        # Class weight matrix: [num_classes * K, embed_dim]
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * K, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=getattr(Config, 'LABEL_SMOOTHING', 0.0))
    
    def forward(self, embeddings, labels):
        # Force float32 for numerical stability (critical for AMP/float16)
        embeddings = embeddings.float()
        weights = F.normalize(self.weight, p=2, dim=1)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Cosine similarity with all sub-centers: [B, num_classes * K]
        cosine_all = F.linear(embeddings, weights)
        
        if self.K > 1:
            # Reshape to [B, num_classes, K] and take max over sub-centers
            cosine_all = cosine_all.view(-1, self.num_classes, self.K)
            cosine = cosine_all.max(dim=2).values  # [B, num_classes]
        else:
            cosine = cosine_all
        
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Arc margin
        sine = torch.sqrt(1.0 - cosine * cosine)
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ + m)
        
        # Numerical stability
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        
        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)
        
        # Apply margin only to the target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        loss = self.criterion(output, labels)
        
        return loss


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     5. LEARNING RATE SCHEDULER                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class PolynomialLRDecay:
    """
    Polynomial LR decay with warmup, as used in AFR-Net (Section 3.5).
    lr = lr_min + (lr_max - lr_min) * (1 - t/T)^power
    """
    
    def __init__(self, optimizer, max_epochs, warmup_epochs=3, lr_min=1e-5, power=3):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.lr_min = lr_min
        self.power = power
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            t = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            lr = self.lr_min + (self.base_lr - self.lr_min) * ((1 - t) ** self.power)
        
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        
        return lr


class CosineAnnealingWithWarmup:
    """Cosine annealing with linear warmup — better for limited data."""
    
    def __init__(self, optimizer, max_epochs, warmup_epochs=5, lr_min=1e-6):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.lr_min = lr_min
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            lr = self.lr_min + (self.base_lr - self.lr_min) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        
        return lr


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    6. TRAINING & EVALUATION LOOP                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss, 
                     best_val_acc, best_train_loss, train_log, config, filepath,
                     arcface_loss=None):
    """Save training state for resume."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'arcface_state_dict': arcface_loss.state_dict() if arcface_loss else None,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'best_train_loss': best_train_loss,
        'train_log': train_log,
        'model_name': config.MODEL_NAME,
        'config': {k: v for k, v in vars(config).items() if k.isupper()},
    }
    torch.save(state, filepath)
    print(f"  💾 Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scaler=None):
    """Load training state for resume."""
    if not os.path.exists(filepath):
        return None
    
    print(f"  📥 Loading checkpoint: {filepath}")
    state = torch.load(filepath, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(state['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    if scaler and state.get('scaler_state_dict'):
        scaler.load_state_dict(state['scaler_state_dict'])
    
    return state


def train_one_epoch(model, train_loader, arcface_loss, optimizer, scaler, config, epoch):
    """Train for one epoch with gradient accumulation and mixed precision."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        
        # Forward pass with mixed precision
        if config.USE_AMP:
            with autocast(dtype=torch.float16):
                embeddings = model(images)
            # ArcFace runs in float32 (handled inside forward)
            loss = arcface_loss(embeddings, labels)
            loss = loss / config.GRAD_ACCUMULATION
        else:
            embeddings = model(images)
            loss = arcface_loss(embeddings, labels)
            loss = loss / config.GRAD_ACCUMULATION
        
        # NaN detection — skip bad batches instead of poisoning training
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            nan_count = getattr(train_one_epoch, '_nan_count', 0) + 1
            train_one_epoch._nan_count = nan_count
            if nan_count <= 5:
                print(f"    ⚠️ NaN/Inf loss at batch {batch_idx+1}, skipping...")
            elif nan_count == 6:
                print(f"    ⚠️ Suppressing further NaN warnings...")
            continue
        
        # Backward pass
        if config.USE_AMP:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % config.GRAD_ACCUMULATION == 0 or (batch_idx + 1) == len(train_loader):
            if config.USE_AMP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
            optimizer.zero_grad()
        
        # Track metrics
        total_loss += loss.item() * config.GRAD_ACCUMULATION
        num_batches += 1
        
        # Accuracy (approximate from ArcFace logits)
        with torch.no_grad():
            emb_norm = F.normalize(embeddings.float(), p=2, dim=1)
            w_norm = F.normalize(arcface_loss.weight, p=2, dim=1)
            logits = F.linear(emb_norm, w_norm)
            # Handle SubCenter ArcFace: collapse K sub-centers per class
            if arcface_loss.K > 1:
                logits = logits.view(logits.size(0), arcface_loss.num_classes, arcface_loss.K)
                logits = logits.max(dim=2).values  # [B, num_classes]
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            speed = (batch_idx + 1) * config.BATCH_SIZE / elapsed
            print(f"    [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {total_loss/num_batches:.4f} | "
                  f"Acc: {100*correct/total:.1f}% | "
                  f"Speed: {speed:.0f} img/s")
    
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, val_loader, arcface_loss, config):
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    
    for images, labels in val_loader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        
        if config.USE_AMP:
            with autocast(dtype=torch.float16):
                embeddings = model(images)
                loss = arcface_loss(embeddings, labels)
        else:
            embeddings = model(images)
            loss = arcface_loss(embeddings, labels)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Accuracy
        emb_norm = F.normalize(embeddings.float(), p=2, dim=1)
        w_norm = F.normalize(arcface_loss.weight, p=2, dim=1)
        logits = F.linear(emb_norm, w_norm)
        # Handle SubCenter ArcFace: collapse K sub-centers per class
        if arcface_loss.K > 1:
            logits = logits.view(logits.size(0), arcface_loss.num_classes, arcface_loss.K)
            logits = logits.max(dim=2).values
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


def train(config=Config):
    """Main training function with full checkpoint support."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ── 1. Build Image List ──
    train_images, val_images = build_image_list()
    
    # ── 2. Assign Subject Labels (with class filtering) ──
    print("\n📋 Assigning subject labels...")
    train_labels, train_sid_map, train_images = assign_numeric_labels(train_images)
    
    # For validation, we need consistent labeling
    # Validation subjects that aren't in training get new labels
    val_labels = []
    val_max_label = len(train_sid_map)
    val_sid_extra = {}
    for fpath, base_path in val_images:
        sid = extract_subject_id(fpath, base_path)
        if sid in train_sid_map:
            val_labels.append(train_sid_map[sid])
        else:
            if sid not in val_sid_extra:
                val_sid_extra[sid] = val_max_label
                val_max_label += 1
            val_labels.append(val_sid_extra[sid])
    
    num_classes = val_max_label
    print(f"  📋 Total classes (train + val): {num_classes:,}")
    
    # ── 3. Create Datasets & Loaders ──
    train_dataset = FingerprintDataset(train_images, train_labels, 
                                       transform=get_transforms(is_training=True),
                                       is_training=True)
    val_dataset = FingerprintDataset(val_images, val_labels,
                                     transform=get_transforms(is_training=False),
                                     is_training=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY, drop_last=True,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE * 2,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
    )
    
    print(f"  🔄 Train batches/epoch: {len(train_loader):,}")
    print(f"  🔄 Val batches/epoch: {len(val_loader):,}")
    
    # ── 4. Build Model ──
    print(f"\n🏗️  Building model: {config.MODEL_NAME}")
    if config.MODEL_NAME == "ms_afrnet":
        model = MSAFRNet(config)
    else:
        model = AFRNet(config)
    
    model = model.to(device)
    total_params, train_params = count_parameters(model)
    print(f"  📊 Total parameters:     {total_params:>12,}")
    print(f"  📊 Trainable parameters: {train_params:>12,}")
    
    # ── 5. Loss, Optimizer, Scheduler ──
    arcface_loss = ArcFaceLoss(
        embed_dim=model.embed_dim,
        num_classes=num_classes,
        margin=config.ARCFACE_MARGIN,
        scale=config.ARCFACE_SCALE,
        K=getattr(config, 'ARCFACE_K', 1),
    ).to(device)
    arcface_k = getattr(config, 'ARCFACE_K', 1)
    print(f"  🎯 ArcFace: s={config.ARCFACE_SCALE}, m={config.ARCFACE_MARGIN}, K={arcface_k} {'(SubCenter)' if arcface_k > 1 else '(Standard)'}")
    
    # Combine model + loss parameters
    all_params = list(model.parameters()) + list(arcface_loss.parameters())
    optimizer = optim.AdamW(all_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    if getattr(config, 'LR_SCHEDULER', 'polynomial') == 'cosine':
        scheduler = CosineAnnealingWithWarmup(
            optimizer, max_epochs=config.MAX_EPOCHS,
            warmup_epochs=config.WARMUP_EPOCHS,
            lr_min=config.LR_MIN,
        )
        print("  📉 Scheduler: Cosine Annealing with Warmup")
    else:
        scheduler = PolynomialLRDecay(
            optimizer, max_epochs=config.MAX_EPOCHS,
            warmup_epochs=config.WARMUP_EPOCHS,
            lr_min=config.LR_MIN, power=config.LR_POWER,
        )
        print("  📉 Scheduler: Polynomial Decay")
    
    scaler = GradScaler() if config.USE_AMP else None
    
    # ── 6. Resume from Checkpoint ──
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{config.MODEL_NAME}_latest.pth")
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_train_loss = float('inf')
    train_log = []
    patience_counter = 0
    
    # Auto-copy checkpoint from uploaded input dataset/model (if provided)
    resume_path = config.RESUME_FROM_INPUT
    if resume_path and os.path.exists(resume_path) and not os.path.exists(checkpoint_path):
        import shutil, glob
        # Handle both file and directory paths
        if os.path.isdir(resume_path):
            # Search for .pth files in the directory
            pth_files = glob.glob(os.path.join(resume_path, "*.pth"))
            if not pth_files:
                pth_files = glob.glob(os.path.join(resume_path, "**/*.pth"), recursive=True)
            if pth_files:
                # Use the first .pth file found (or one matching model name)
                src = pth_files[0]
                for f in pth_files:
                    if config.MODEL_NAME in os.path.basename(f):
                        src = f
                        break
                shutil.copy2(src, checkpoint_path)
                print(f"  📋 Copied checkpoint from: {src}")
            else:
                # Maybe the files don't have .pth extension — copy whatever is there
                all_files = [f for f in os.listdir(resume_path) if os.path.isfile(os.path.join(resume_path, f))]
                if all_files:
                    src = os.path.join(resume_path, all_files[0])
                    shutil.copy2(src, checkpoint_path)
                    print(f"  📋 Copied checkpoint from: {src}")
                else:
                    print(f"  ⚠️ No checkpoint files found in: {resume_path}")
        else:
            shutil.copy2(resume_path, checkpoint_path)
            print(f"  📋 Copied checkpoint from: {resume_path}")
    
    # Try loading checkpoint
    if os.path.exists(checkpoint_path):
        print(f"  📥 Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        
        # Load model weights
        model.load_state_dict(state['model_state_dict'])
        print(f"  ✅ Model weights loaded")
        
        # Load ArcFace weights (if saved and size matches)
        if state.get('arcface_state_dict'):
            ckpt_arcface_size = state['arcface_state_dict']['weight'].shape[0]
            if ckpt_arcface_size == num_classes:
                arcface_loss.load_state_dict(state['arcface_state_dict'])
                print(f"  ✅ ArcFace weights loaded ({num_classes} classes)")
            else:
                print(f"  ⚠️ ArcFace size mismatch: checkpoint={ckpt_arcface_size}, current={num_classes}")
                print(f"  ⚠️ ArcFace weights will start fresh (model weights are fine)")
        
        # Load optimizer + scaler
        try:
            optimizer.load_state_dict(state['optimizer_state_dict'])
            if scaler and state.get('scaler_state_dict'):
                scaler.load_state_dict(state['scaler_state_dict'])
            print(f"  ✅ Optimizer state loaded")
        except Exception as e:
            print(f"  ⚠️ Optimizer state mismatch, restarting optimizer: {str(e)[:80]}")
            all_params = list(model.parameters()) + list(arcface_loss.parameters())
            optimizer = optim.AdamW(all_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
            scheduler = PolynomialLRDecay(
                optimizer, max_epochs=config.MAX_EPOCHS,
                warmup_epochs=config.WARMUP_EPOCHS,
                lr_min=config.LR_MIN, power=config.LR_POWER,
            )
        
        start_epoch = state['epoch'] + 1
        best_val_loss = state.get('best_val_loss', float('inf'))
        best_val_acc = state.get('best_val_acc', 0.0)
        best_train_loss = state.get('best_train_loss', float('inf'))
        train_log = state.get('train_log', [])
        print(f"  ✅ Resumed from epoch {start_epoch}, best_train_loss={best_train_loss:.4f}")
    
    # ── 7. Training Loop ──
    print(f"\n{'='*60}")
    print(f"  🚀 TRAINING: {config.MODEL_NAME.upper()}")
    print(f"     Epochs {start_epoch} → {config.MAX_EPOCHS}")
    print(f"     Effective batch: {config.BATCH_SIZE * config.GRAD_ACCUMULATION}")
    print(f"{'='*60}\n")
    
    training_start = time.time()
    
    for epoch in range(start_epoch, config.MAX_EPOCHS):
        epoch_start = time.time()
        
        # Update learning rate
        current_lr = scheduler.step(epoch)
        
        print(f"\n📌 Epoch {epoch+1}/{config.MAX_EPOCHS} | LR: {current_lr:.2e}")
        print(f"{'─'*50}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, arcface_loss, optimizer, scaler, config, epoch
        )
        
        epoch_time = time.time() - epoch_start
        
        print(f"  📈 Train | Loss: {train_loss:.4f} | Acc: {100*train_acc:.2f}% | Time: {epoch_time:.0f}s")
        
        # Validate
        val_loss, val_acc = 0.0, 0.0
        if (epoch + 1) % config.VAL_EVERY_N_EPOCHS == 0 and len(val_loader) > 0:
            val_loss, val_acc = validate(model, val_loader, arcface_loss, config)
            print(f"  📉 Val   | Loss: {val_loss:.4f} | Acc: {100*val_acc:.2f}%")
        
        # Log
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr,
            'time_s': epoch_time,
        }
        train_log.append(log_entry)
        
        # Save best model (based on training loss since val subjects are disjoint)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            patience_counter = 0
            best_path = os.path.join(config.CHECKPOINT_DIR, f"{config.MODEL_NAME}_best.pth")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, 
                          best_val_loss, best_val_acc, best_train_loss, train_log, config, best_path,
                          arcface_loss=arcface_loss)
            print(f"  🏆 New best! Train Loss: {best_train_loss:.4f}")
        else:
            patience_counter += 1
        
        # Track best val separately (for logging)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
        
        # Save periodic checkpoint
        if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                          best_val_loss, best_val_acc, best_train_loss, train_log, config, checkpoint_path,
                          arcface_loss=arcface_loss)
        
        # Save training log CSV
        log_path = os.path.join(config.LOG_DIR, f"{config.MODEL_NAME}_training_log.csv")
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            writer.writeheader()
            writer.writerows(train_log)
        
        # Early stopping (on train loss — val subjects are disjoint so val_acc is unreliable)
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch+1} (train loss didn't improve for {config.EARLY_STOPPING_PATIENCE} epochs)")
            break
        
        # Time check — save and stop if close to Kaggle limit
        total_elapsed = time.time() - training_start
        avg_epoch_time = total_elapsed / (epoch - start_epoch + 1)
        remaining_time = 12 * 3600 - total_elapsed  # 12 hours in seconds
        
        if remaining_time < avg_epoch_time * 1.5:
            print(f"\n⏰ Approaching Kaggle time limit! Saving checkpoint...")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                          best_val_loss, best_val_acc, best_train_loss, train_log, config, checkpoint_path,
                          arcface_loss=arcface_loss)
            print(f"  📊 Completed {epoch - start_epoch + 1} epochs in this session")
            print(f"  📊 Best Val Acc: {100*best_val_acc:.2f}%")
            print(f"  💡 To resume: set same MODEL_NAME and re-run this notebook")
            break
    
    # Final save
    save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                  best_val_loss, best_val_acc, best_train_loss, train_log, config, checkpoint_path,
                  arcface_loss=arcface_loss)
    
    total_time = time.time() - training_start
    print(f"\n{'='*60}")
    print(f"  ✅ TRAINING COMPLETE")
    print(f"     Total time: {total_time/3600:.1f} hours")
    print(f"     Best Val Acc: {100*best_val_acc:.2f}%")
    print(f"     Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    return model, train_log


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           7. RUN TRAINING                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # ── Step 1: Visualize Preprocessing (run once) ──
    visualize_preprocessing(num_samples=3, save_dir="/kaggle/working")
    
    # ── Step 2: Fingerprint Type Gallery ──
    visualize_fingerprint_types(save_dir="/kaggle/working")
    
    # ╔════════════════════════════════════════════════════════════╗
    # ║  UNCOMMENT ONE BLOCK BELOW — AFR-Net OR MS-AFR-Net       ║
    # ╚════════════════════════════════════════════════════════════╝
    
    # ──────────────────────────────────────────────────────────────
    # OPTION A: AFR-Net Baseline (improved training)
    # ──────────────────────────────────────────────────────────────
    # Config.MODEL_NAME = "afrnet"
    # Config.RESUME_FROM_INPUT = None
    # Config.ARCFACE_SCALE = 64
    # Config.ARCFACE_MARGIN = 0.5
    # Config.LEARNING_RATE = 1e-4
    # Config.WEIGHT_DECAY = 5e-5
    # Config.WARMUP_EPOCHS = 5
    # Config.MAX_EPOCHS = 75
    # Config.LR_SCHEDULER = "polynomial"
    # Config.LR_POWER = 3
    # Config.LR_MIN = 1e-5
    # Config.MIN_IMAGES_PER_CLASS = 5
    # Config.LABEL_SMOOTHING = 0.0
    # Config.BATCH_SIZE = 64
    # Config.GRAD_ACCUMULATION = 1
    # Config.NUM_WORKERS = 4
    # Config.DEBUG = False
    
    # ──────────────────────────────────────────────────────────────
    # OPTION B: MS-AFR-Net — MAXIMUM PERFORMANCE (403K data)
    # ──────────────────────────────────────────────────────────────
    Config.MODEL_NAME = "msafrnet"
    Config.RESUME_FROM_INPUT = None
    
    # ArcFace — s=64, m=0.5 is optimal (paper-proven)
    Config.ARCFACE_SCALE = 64
    Config.ARCFACE_MARGIN = 0.5
    Config.ARCFACE_K = 3                # SubCenter ArcFace: 3 sub-centers per class
    
    # Label smoothing: prevents overconfidence on small classes
    Config.LABEL_SMOOTHING = 0.1
    
    # Cosine annealing: decays LR smoothly, better than polynomial for limited data
    Config.LR_SCHEDULER = "cosine"
    Config.LEARNING_RATE = 2e-4         # Slightly higher (cosine can handle it)
    Config.WEIGHT_DECAY = 1e-4          # Strong regularization for 403K
    Config.WARMUP_EPOCHS = 5            # Gentle warmup
    Config.MAX_EPOCHS = 75              # Full training
    Config.LR_MIN = 1e-6               # Near-zero floor
    
    # Class filtering: drop under-represented classes
    Config.MIN_IMAGES_PER_CLASS = 5
    
    # Hardware
    Config.BATCH_SIZE = 64
    Config.GRAD_ACCUMULATION = 1
    Config.NUM_WORKERS = 4
    
    Config.DEBUG = False
    
    # ── Run ──
    model, log = train(Config)
"""

╔══════════════════════════════════════════════════════════════════════════════╗
║                         USAGE INSTRUCTIONS                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  PHASE 1 (Session 1): Train AFR-Net Baseline                              ║
║    1. Set Config.MODEL_NAME = "afrnet"                                     ║
║    2. Set Config.DEBUG = True  (test first!)                               ║  
║    3. Run notebook → verify no errors                                      ║
║    4. Set Config.DEBUG = False                                             ║
║    5. Run again → trains until time limit, auto-saves checkpoint           ║
║                                                                            ║
║  PHASE 2 (Session 2): Resume or Train MS-AFR-Net                          ║
║    Option A: Resume AFR-Net                                                ║
║      - Keep MODEL_NAME = "afrnet", re-run → auto-resumes from checkpoint  ║
║    Option B: Start MS-AFR-Net                                              ║
║      - Change MODEL_NAME = "ms_afrnet", re-run                            ║
║                                                                            ║
║  PHASE 3 (Session 3): Continue + Evaluate                                  ║
║    - Resume whichever model didn't finish                                  ║
║    - Run evaluation notebook separately                                    ║
║                                                                            ║
║  CHECKPOINTS: Saved to /kaggle/working/checkpoints/                       ║
║    - {model}_latest.pth — auto-resume point                               ║
║    - {model}_best.pth   — best validation accuracy                        ║
║                                                                            ║
║  IMPORTANT: Download checkpoints from /kaggle/working/ after each         ║
║  session!  Re-upload them to the next session's input to resume.          ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
