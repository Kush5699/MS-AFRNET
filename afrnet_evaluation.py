"""
MS-AFR-Net Evaluation — All-in-One Script
Extracts embeddings + computes metrics + generates plots.

Kaggle Inputs Required:
  Model:  talakshrabari/ms-afrnet-best-version2  (checkpoint)
  Data1:  kushp3690/molf-testing-dataset
  Data2:  kushp3690/polyu-testing-datasets
  Data3:  kushp3690/ispfdv2-testing-dataset
  Data4:  kushp3690/afr-training-real      (FVC + SD302 test splits)
"""

MODEL_NAME = "MS-AFR-Net"

import os, sys, time, math, random, json, warnings, glob
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from torch.amp import autocast  # PyTorch 2.x
except ImportError:
    from torch.cuda.amp import autocast  # PyTorch 1.x
import torchvision.transforms as T
import torchvision.models as models

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════════════════════════
#  1. PATHS
# ═══════════════════════════════════════════════════════════════

CHECKPOINT_DIR = "/kaggle/input/models/talakshrabari/ms-afrnet-best-version2/pytorch/default/1"
MOLF_PATH      = "/kaggle/input/datasets/kushp3690/molf-testing-dataset"
POLYU_PATH     = "/kaggle/input/datasets/kushp3690/polyu-testing-datasets"
ISPFDV2_PATH   = "/kaggle/input/datasets/kushp3690/ispfdv2-testing-dataset"
FVC_SD302_PATH = "/kaggle/input/datasets/kushp3690/afr-training-real"
OUTPUT_DIR     = "/kaggle/working"

INPUT_SIZE     = 224
EMBEDDING_DIM  = 384
BATCH_SIZE     = 64
NUM_WORKERS    = 2
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


# ═══════════════════════════════════════════════════════════════
#  2. MODEL ARCHITECTURE  (must match MS-AFR-Net training exactly)
# ═══════════════════════════════════════════════════════════════

class SpatialTransformerNetwork(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(True), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(True), nn.Dropout(0.1), nn.Linear(64, 6),
        )
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))

    def forward(self, x):
        B = x.size(0)
        f = self.localization(x).view(B, -1)
        theta = self.fc(f).view(B, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='border')


class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=False, freeze_stages=0, gradient_checkpointing=False):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.conv1  = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


class CNNHead(nn.Module):
    def __init__(self, in_channels=2048, embed_dim=384):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(nn.Linear(in_channels, embed_dim), nn.BatchNorm1d(embed_dim))

    def forward(self, c5):
        x = self.gap(c5).view(c5.size(0), -1)
        return self.fc(x)


class MultiScaleFeaturePyramid(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.proj3 = nn.Sequential(nn.Conv2d(512,  embed_dim, 1), nn.BatchNorm2d(embed_dim), nn.GELU())
        self.proj4 = nn.Sequential(nn.Conv2d(1024, embed_dim, 1), nn.BatchNorm2d(embed_dim), nn.GELU())
        self.proj5 = nn.Sequential(nn.Conv2d(2048, embed_dim, 1), nn.BatchNorm2d(embed_dim), nn.GELU())
        self.scale_embed_3 = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.scale_embed_4 = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.scale_embed_5 = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_3 = nn.Parameter(torch.randn(1, 784, embed_dim) * 0.02)
        self.pos_4 = nn.Parameter(torch.randn(1, 196, embed_dim) * 0.02)
        self.pos_5 = nn.Parameter(torch.randn(1,  49, embed_dim) * 0.02)

    def forward(self, c3, c4, c5):
        f3 = self.proj3(c3).flatten(2).transpose(1,2) + self.pos_3 + self.scale_embed_3
        f4 = self.proj4(c4).flatten(2).transpose(1,2) + self.pos_4 + self.scale_embed_4
        f5 = self.proj5(c5).flatten(2).transpose(1,2) + self.pos_5 + self.scale_embed_5
        return torch.cat([f3, f4, f5], dim=1), (f3.size(1), f4.size(1), f5.size(1))


class CrossScaleAttention(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, num_layers=6, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        B = tokens.size(0)
        x = torch.cat([self.cls_token.expand(B,-1,-1), tokens], dim=1)
        x = self.transformer(x)
        return self.norm(x[:, 0]), x[:, 1:]


class AdaptiveScaleFusionGate(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.scale_proj = nn.Sequential(nn.Linear(embed_dim*4, embed_dim), nn.GELU(), nn.Dropout(0.1))
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim//2), nn.GELU(), nn.Linear(embed_dim//2, 3), nn.Softmax(dim=-1))
        self.output_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim))

    def forward(self, cls_out, token_out, scale_lengths):
        n3, n4, n5 = scale_lengths
        p3 = token_out[:, :n3].mean(1)
        p4 = token_out[:, n3:n3+n4].mean(1)
        p5 = token_out[:, n3+n4:n3+n4+n5].mean(1)
        combined = self.scale_proj(torch.cat([cls_out, p3, p4, p5], -1))
        g = self.gate(combined)
        z = g[:,0:1]*p3 + g[:,1:2]*p4 + g[:,2:3]*p5
        return self.output_proj(z), g


class MSAFRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stn = SpatialTransformerNetwork(3)
        self.backbone = ResNet50Backbone(pretrained=False, freeze_stages=0, gradient_checkpointing=False)
        self.cnn_head = CNNHead(2048, EMBEDDING_DIM)
        self.feature_pyramid = MultiScaleFeaturePyramid(EMBEDDING_DIM)
        self.cross_attention = CrossScaleAttention(EMBEDDING_DIM, 6, 6, 0.1)
        self.fusion_gate = AdaptiveScaleFusionGate(EMBEDDING_DIM)
        self.embed_dim = EMBEDDING_DIM * 2

    def forward(self, x):
        x = self.stn(x)
        c3, c4, c5 = self.backbone(x)
        z_c = self.cnn_head(c5)
        ms_tokens, sl = self.feature_pyramid(c3, c4, c5)
        cls_out, tok_out = self.cross_attention(ms_tokens)
        z_ms, _ = self.fusion_gate(cls_out, tok_out, sl)
        return torch.cat([z_c, z_ms], dim=-1)


# ═══════════════════════════════════════════════════════════════
#  3. LOAD CHECKPOINT
# ═══════════════════════════════════════════════════════════════

def load_model():
    model = MSAFRNet().to(DEVICE)
    # Find checkpoint file
    ckpt_path = CHECKPOINT_DIR
    if os.path.isdir(ckpt_path):
        pth = glob.glob(os.path.join(ckpt_path, "**/*.pth"), recursive=True)
        if not pth:
            pth = [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path)
                   if os.path.isfile(os.path.join(ckpt_path, f))]
        ckpt_path = pth[0] if pth else ckpt_path

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    
    if 'model_state_dict' in state:
        sd = state['model_state_dict']
        epoch = state.get('epoch', '?')
        print(f"  Checkpoint from epoch {epoch}")
    else:
        sd = state
    
    # Clean state dict keys
    cleaned = {}
    for k, v in sd.items():
        k = k.replace('module.', '')
        cleaned[k] = v
    
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  ⚠️ Missing keys: {len(missing)} (expected for eval-only model)")
        for mk in missing[:5]:
            print(f"      {mk}")
    if unexpected:
        print(f"  ⚠️ Unexpected keys: {len(unexpected)} (ignored)")
        for uk in unexpected[:5]:
            print(f"      {uk}")
    
    model.eval()
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_p:,}")
    print(f"  Model loaded successfully ✅")
    return model

model = load_model()


# ═══════════════════════════════════════════════════════════════
#  4. IMAGE UTILITIES
# ═══════════════════════════════════════════════════════════════

def safe_open(fpath):
    try:
        img = Image.open(fpath)
        if img.width < 32 or img.height < 32:
            return None
        m = img.mode
        if m in ('I;16', 'I'):
            a = np.array(img, dtype=np.float32)
            if a.max() > 0: a = (a / a.max() * 255).astype(np.uint8)
            else: a = a.astype(np.uint8)
            img = Image.fromarray(a, 'L').convert('RGB')
        elif m == 'RGBA':
            bg = Image.new('RGB', img.size, (255,255,255))
            bg.paste(img, mask=img.split()[3]); img = bg
        elif m != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception:
        return None

eval_transform = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


class EvalDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = safe_open(self.files[idx])
        if img is None:
            img = Image.new('RGB', (INPUT_SIZE, INPUT_SIZE), (0,0,0))
        return eval_transform(img), idx


# ═══════════════════════════════════════════════════════════════
#  5. EMBEDDING EXTRACTION
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(model, file_list, batch_size=BATCH_SIZE):
    """Extract 768-d embeddings with TTA (original + horizontal flip, averaged)."""
    ds = EvalDataset(file_list)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)
    all_emb = []
    for imgs, _ in dl:
        imgs = imgs.to(DEVICE)
        with autocast(device_type='cuda', dtype=torch.float16):
            # Original
            emb_orig = model(imgs)
            # Horizontal flip
            emb_flip = model(torch.flip(imgs, dims=[3]))
        # Average both embeddings
        emb = (emb_orig.float() + emb_flip.float()) / 2.0
        emb = F.normalize(emb, p=2, dim=1)
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


# ═══════════════════════════════════════════════════════════════
#  6. DATASET SCANNERS
# ═══════════════════════════════════════════════════════════════

def scan_images(root):
    imgs = []
    for r, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                imgs.append(os.path.join(r, f))
    return sorted(imgs)


def scan_molf(base):
    """MOLF: DB1(optical), DB2(capacitive), DB3(contactless), DB4(latent)"""
    result = {}
    for db_name in ['DB1', 'DB2', 'DB3', 'DB4']:
        db_path = None
        for r, dirs, _ in os.walk(base):
            for d in dirs:
                if d.upper() == db_name:
                    db_path = os.path.join(r, d)
                    break
            if db_path: break
        if not db_path:
            # try case-insensitive
            for r, dirs, _ in os.walk(base):
                for d in dirs:
                    if d.lower() == db_name.lower():
                        db_path = os.path.join(r, d)
                        break
                if db_path: break
        if db_path:
            imgs = scan_images(db_path)
            # Extract subject IDs from filename: typically "XXXX_Y_Z.ext" -> subject=XXXX
            subjects = {}
            for fp in imgs:
                fname = Path(fp).stem
                parts = fname.split('_')
                sid = parts[0] if parts else fname[:4]
                subjects.setdefault(sid, []).append(fp)
            result[db_name] = {'path': db_path, 'images': imgs, 'subjects': subjects}
            print(f"  MOLF {db_name}: {len(imgs)} images, {len(subjects)} subjects")
    return result


def scan_polyu(base):
    """PolyU: contactless vs contact-based fingerprints"""
    result = {}
    for r, dirs, _ in os.walk(base):
        for d in dirs:
            full = os.path.join(r, d)
            imgs = scan_images(full)
            if len(imgs) > 0:
                subjects = {}
                for fp in imgs:
                    fname = Path(fp).stem
                    parts = fname.replace('-', '_').split('_')
                    sid = parts[0] if parts else fname[:4]
                    subjects.setdefault(sid, []).append(fp)
                result[d] = {'path': full, 'images': imgs, 'subjects': subjects}
                print(f"  PolyU/{d}: {len(imgs)} images, {len(subjects)} subjects")
        break  # only top-level dirs
    return result


def scan_ispfdv2(base):
    """ISPFDv2: smartphone fingerphoto datasets"""
    result = {}
    imgs = scan_images(base)
    subjects = {}
    for fp in imgs:
        fname = Path(fp).stem
        parts = fname.split('_')
        sid = parts[0] if parts else fname[:4]
        subjects.setdefault(sid, []).append(fp)
    result['ISPFDv2'] = {'path': base, 'images': imgs, 'subjects': subjects}
    print(f"  ISPFDv2: {len(imgs)} images, {len(subjects)} subjects")
    return result


def scan_fvc_test(base):
    """FVC 2002/2004 DB1A-DB3A test subsets"""
    fvc_test_patterns = {
        'FVC2002_DB1A': ['fvc2002', 'db1_a', 'db1a', 'db1 a'],
        'FVC2002_DB2A': ['fvc2002', 'db2_a', 'db2a', 'db2 a'],
        'FVC2002_DB3A': ['fvc2002', 'db3_a', 'db3a', 'db3 a'],
        'FVC2004_DB1A': ['fvc2004', 'db1_a', 'db1a', 'db1 a'],
        'FVC2004_DB2A': ['fvc2004', 'db2_a', 'db2a', 'db2 a'],
        'FVC2004_DB3A': ['fvc2004', 'db3_a', 'db3a', 'db3 a'],
    }
    result = {}
    all_imgs = scan_images(base)
    for name, patterns in fvc_test_patterns.items():
        year_pat = patterns[0]
        db_pats = patterns[1:]
        matched = []
        for fp in all_imgs:
            fp_low = fp.lower().replace('\\', '/')
            if year_pat in fp_low:
                for dp in db_pats:
                    if dp in fp_low.replace('_', '').replace(' ', ''):
                        matched.append(fp)
                        break
        if not matched:
            # Try directory-name matching
            for fp in all_imgs:
                fp_low = fp.lower().replace('\\', '/')
                if year_pat in fp_low and any(dp in fp_low for dp in db_pats):
                    matched.append(fp)
        if matched:
            subjects = {}
            for fp in matched:
                fname = Path(fp).stem
                sid = fname.split('_')[0] if '_' in fname else fname[:3]
                subjects.setdefault(sid, []).append(fp)
            result[name] = {'images': matched, 'subjects': subjects}
            print(f"  {name}: {len(matched)} images, {len(subjects)} subjects")
    return result


def scan_sd302_test(base, seed=42):
    """NIST SD302: take the test split (last 10% of subjects)"""
    all_imgs = scan_images(base)
    sd302_imgs = [f for f in all_imgs if 'sd302' in f.lower().replace('\\', '/')]
    if not sd302_imgs:
        print("  SD302: not found in dataset")
        return {}

    subjects = {}
    for fp in sd302_imgs:
        fname = os.path.basename(fp)
        sid = fname.split('_')[0] if '_' in fname else fname[:8]
        subjects.setdefault(sid, []).append(fp)

    sorted_subs = sorted(subjects.keys())
    rng = random.Random(seed)
    rng.shuffle(sorted_subs)
    n = len(sorted_subs)
    test_subs = set(sorted_subs[int(n*0.9):])  # last 10%

    test_imgs = []
    test_subjects = {}
    for s in test_subs:
        test_subjects[s] = subjects[s]
        test_imgs.extend(subjects[s])

    result = {'SD302': {'images': test_imgs, 'subjects': test_subjects}}
    print(f"  SD302 test split: {len(test_imgs)} images, {len(test_subjects)} subjects")
    return result


# ═══════════════════════════════════════════════════════════════
#  7. SCAN ALL DATASETS + EXTRACT EMBEDDINGS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  SCANNING TEST DATASETS")
print("="*60)

datasets = {}

print("\n[1/4] MOLF...")
if os.path.exists(MOLF_PATH):
    datasets['MOLF'] = scan_molf(MOLF_PATH)

print("\n[2/4] PolyU...")
if os.path.exists(POLYU_PATH):
    datasets['PolyU'] = scan_polyu(POLYU_PATH)

print("\n[3/4] ISPFDv2...")
if os.path.exists(ISPFDV2_PATH):
    datasets['ISPFDv2'] = scan_ispfdv2(ISPFDV2_PATH)

print("\n[4/4] FVC + SD302...")
if os.path.exists(FVC_SD302_PATH):
    datasets['FVC'] = scan_fvc_test(FVC_SD302_PATH)
    datasets['SD302'] = scan_sd302_test(FVC_SD302_PATH)

# Save dataset structure info
ds_info = {}
for group_name, group_data in datasets.items():
    ds_info[group_name] = {}
    for subset_name, subset_data in group_data.items():
        ds_info[group_name][subset_name] = {
            'num_images': len(subset_data['images']),
            'num_subjects': len(subset_data['subjects']),
        }
info_path = os.path.join(OUTPUT_DIR, "dataset_info.json")
with open(info_path, 'w') as f:
    json.dump(ds_info, f, indent=2)
print(f"\nDataset info saved to {info_path}")

# ── Extract embeddings for each subset ──
print("\n" + "="*60)
print("  EXTRACTING EMBEDDINGS")
print("="*60)

for group_name, group_data in datasets.items():
    for subset_name, subset_data in group_data.items():
        imgs = subset_data['images']
        if len(imgs) == 0:
            continue
        safe_name = f"{group_name}_{subset_name}".replace('/', '_')
        print(f"\n  {group_name}/{subset_name}: {len(imgs)} images...")

        # Extract AFR-Net embeddings (768-d: z_c + z_a)
        t0 = time.time()
        emb = extract_embeddings(model, imgs)
        print(f"    AFR-Net (768-d): {emb.shape} in {time.time()-t0:.1f}s")
        np.save(os.path.join(OUTPUT_DIR, f"emb_{safe_name}.npy"), emb)

        # Save file list + subject mapping
        file_subjects = []
        subjects_dict = subset_data['subjects']
        # Build reverse mapping: filepath -> subject_id
        fp_to_sid = {}
        for sid, fps in subjects_dict.items():
            for fp in fps:
                fp_to_sid[fp] = sid
        for fp in imgs:
            file_subjects.append({
                'filepath': fp,
                'subject_id': fp_to_sid.get(fp, Path(fp).stem[:4])
            })
        with open(os.path.join(OUTPUT_DIR, f"meta_{safe_name}.json"), 'w') as f:
            json.dump(file_subjects, f)

print("\n" + "="*60)
print("  EMBEDDING EXTRACTION COMPLETE")
print("="*60)


# ═══════════════════════════════════════════════════════════════
#  8. BIOMETRIC METRIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_score_matrix(embeddings):
    return embeddings @ embeddings.T

def generate_genuine_impostor_scores(embeddings, subject_ids):
    n = len(embeddings)
    sim_matrix = compute_score_matrix(embeddings)
    genuine, impostor = [], []
    subj_to_idx = defaultdict(list)
    for i, sid in enumerate(subject_ids):
        subj_to_idx[sid].append(i)
    for sid, indices in subj_to_idx.items():
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                genuine.append(sim_matrix[indices[i], indices[j]])
    all_subs = list(subj_to_idx.keys())
    max_impostor = min(len(genuine) * 10, 500000)
    rng = np.random.RandomState(42)
    count = 0
    for _ in range(max_impostor * 3):
        if count >= max_impostor: break
        s1, s2 = rng.choice(len(all_subs), 2, replace=False)
        i = rng.choice(subj_to_idx[all_subs[s1]])
        j = rng.choice(subj_to_idx[all_subs[s2]])
        impostor.append(sim_matrix[i, j])
        count += 1
    return np.array(genuine), np.array(impostor)

def compute_eer(genuine, impostor):
    thresholds = np.linspace(min(impostor.min(), genuine.min()),
                             max(impostor.max(), genuine.max()), 10000)
    far = np.array([np.mean(impostor >= t) for t in thresholds])
    frr = np.array([np.mean(genuine < t) for t in thresholds])
    diffs = far - frr
    idx = np.argmin(np.abs(diffs))
    eer = (far[idx] + frr[idx]) / 2.0
    return eer, thresholds[idx], thresholds, far, frr

def compute_tar_at_far(genuine, impostor, target_far=0.001):
    # Sort impostor scores descending — use them directly as thresholds
    # This gives exact FAR values at each impostor score
    sorted_imp = np.sort(impostor)[::-1]
    n_imp = len(sorted_imp)
    # Find the threshold where FAR = target_far
    # FAR = (number of impostors >= threshold) / total impostors
    # At index k: FAR = (k+1) / n_imp
    target_idx = int(np.floor(target_far * n_imp)) - 1
    if target_idx < 0:
        target_idx = 0
    threshold = sorted_imp[target_idx]
    actual_far = np.mean(impostor >= threshold)
    # If actual FAR exceeds target, move threshold up
    while actual_far > target_far and target_idx > 0:
        target_idx -= 1
        threshold = sorted_imp[target_idx]
        actual_far = np.mean(impostor >= threshold)
    tar = np.mean(genuine >= threshold)
    return tar, actual_far, threshold

def compute_cmc(gallery_emb, gallery_ids, probe_emb, probe_ids):
    sim = probe_emb @ gallery_emb.T
    ranks = []
    for i in range(len(probe_emb)):
        sorted_idx = np.argsort(-sim[i])
        true_id = probe_ids[i]
        for rank, idx in enumerate(sorted_idx):
            if gallery_ids[idx] == true_id:
                ranks.append(rank + 1)
                break
        else:
            ranks.append(len(gallery_emb) + 1)
    ranks = np.array(ranks)
    max_rank = min(len(gallery_emb), 100)
    cmc = np.zeros(max_rank)
    for r in range(max_rank):
        cmc[r] = np.mean(ranks <= (r + 1))
    return cmc, ranks


# ═══════════════════════════════════════════════════════════════
#  9. TEST DATASET FINGERPRINT TYPES
# ═══════════════════════════════════════════════════════════════

TEST_DATASET_INFO = {
    'FVC_FVC2002_DB1A': ('Live — Optical',       'KeyTronic FCD4B14',        100),
    'FVC_FVC2002_DB2A': ('Live — Optical',       'Biometrika FX2000',        100),
    'FVC_FVC2002_DB3A': ('Live — Capacitive',    'Precise Biometrics MC100', 100),
    'FVC_FVC2004_DB1A': ('Live — Optical',       'CrossMatch V300',          100),
    'FVC_FVC2004_DB2A': ('Live — Optical',       'Digital Persona U.are.U',  100),
    'FVC_FVC2004_DB3A': ('Live — Thermal Sweep', 'Atmel FingerChip FCD4B14', 100),
    'ISPFDv2_ISPFDv2':  ('Fingerphoto',          'Smartphone camera',         75),
    'PolyU_contact-based_fingerprints':                  ('Live — Optical',    'Contact scanner',  336),
    'PolyU_contactless_2d_fingerprint_images':            ('Contactless 2D',   'Camera capture',     6),
    'PolyU_processed_contactless_2d_fingerprint_images':  ('Processed Contactless', 'Camera + processing', 6),
    'SD302_SD302':      ('Mixed (Rolled+Latent)', 'Crossmatch + Crime-scene', 20),
}

print("\n" + "="*80)
print("  EVALUATION DATASETS — FINGERPRINT TYPE BREAKDOWN")
print("="*80)
print(f"  {'Dataset':<50} {'Type':<25} {'Sensor'}")
print("-"*80)
for dname, (ftype, sensor, nsub) in TEST_DATASET_INFO.items():
    print(f"  {dname:<50} {ftype:<25} {sensor} ({nsub} subj)")
print("="*80)


# ═══════════════════════════════════════════════════════════════
#  10. LOAD EMBEDDINGS & COMPUTE METRICS
# ═══════════════════════════════════════════════════════════════

def load_all_embeddings(input_dir=OUTPUT_DIR, prefix='emb_'):
    data = {}
    for f in sorted(os.listdir(input_dir)):
        if f.startswith(prefix) and f.endswith(".npy"):
            key = f[len(prefix):-4]
            emb_arr = np.load(os.path.join(input_dir, f))
            meta_file = os.path.join(input_dir, f"meta_{key}.json")
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as mf:
                    meta = json.load(mf)
            else:
                meta = [{'filepath': f'img_{i}', 'subject_id': str(i)} for i in range(len(emb_arr))]
            data[key] = {'embeddings': emb_arr, 'meta': meta}
            print(f"  Loaded {key}: {emb_arr.shape}")
    return data

print(f"\nLoading {MODEL_NAME} embeddings...")
all_data = load_all_embeddings()

if not all_data:
    print("ERROR: No embeddings found!")
    sys.exit(1)

results = {}
all_genuine = {}
all_impostor = {}

print("\n" + "="*60)
print("  COMPUTING BIOMETRIC METRICS")
print("="*60)

for key, dset in all_data.items():
    emb_data = dset['embeddings']
    meta = dset['meta']
    sids = [m['subject_id'] for m in meta]
    unique_subs = set(sids)
    subj_counts = defaultdict(int)
    for s in sids:
        subj_counts[s] += 1
    multi_sample_subs = {s for s, c in subj_counts.items() if c >= 2}
    if len(multi_sample_subs) < 2:
        print(f"\n  {key}: skipping (need >= 2 subjects with >= 2 samples each)")
        continue

    print(f"\n  {key}: {len(emb_data)} images, {len(unique_subs)} subjects "
          f"({len(multi_sample_subs)} with 2+ samples)")
    if key in TEST_DATASET_INFO:
        ftype, sensor, _ = TEST_DATASET_INFO[key]
        print(f"    Type: {ftype}  |  Sensor: {sensor}")

    genuine, impostor = generate_genuine_impostor_scores(emb_data, sids)
    print(f"    Genuine pairs: {len(genuine):,}, Impostor pairs: {len(impostor):,}")
    if len(genuine) < 5 or len(impostor) < 5:
        print(f"    Too few pairs, skipping")
        continue

    eer, eer_thresh, thresholds, far_curve, frr_curve = compute_eer(genuine, impostor)
    tar_01, _, _ = compute_tar_at_far(genuine, impostor, 0.001)
    tar_001, _, _ = compute_tar_at_far(genuine, impostor, 0.0001)

    print(f"    EER:            {eer*100:.2f}%")
    print(f"    TAR@FAR=0.1%:   {tar_01:.4f}")
    print(f"    TAR@FAR=0.01%:  {tar_001:.4f}")

    # Identification (CMC)
    gallery_idx, probe_idx = [], []
    seen = set()
    for i, s in enumerate(sids):
        if s not in seen and s in multi_sample_subs:
            gallery_idx.append(i)
            seen.add(s)
        elif s in multi_sample_subs:
            probe_idx.append(i)

    rank1, rank5, cmc = 0.0, 0.0, None
    if len(gallery_idx) >= 2 and len(probe_idx) >= 2:
        g_emb = emb_data[gallery_idx]
        g_ids = [sids[i] for i in gallery_idx]
        p_emb = emb_data[probe_idx]
        p_ids = [sids[i] for i in probe_idx]
        cmc, ranks = compute_cmc(g_emb, g_ids, p_emb, p_ids)
        rank1 = cmc[0] if len(cmc) > 0 else 0
        rank5 = cmc[4] if len(cmc) > 4 else 0
        print(f"    Rank-1:         {rank1*100:.2f}% (gallery={len(g_ids)}, probe={len(p_ids)})")
        print(f"    Rank-5:         {rank5*100:.2f}%")

    results[key] = {
        'eer': eer, 'eer_threshold': eer_thresh,
        'tar_far01': tar_01, 'tar_far001': tar_001,
        'rank1': rank1, 'rank5': rank5,
        'n_images': len(emb_data), 'n_subjects': len(unique_subs),
        'n_genuine': len(genuine), 'n_impostor': len(impostor),
    }
    all_genuine[key] = genuine
    all_impostor[key] = impostor
    if cmc is not None:
        results[key]['cmc'] = cmc.tolist()


# ═══════════════════════════════════════════════════════════════
#  10. PLOTS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  GENERATING PLOTS")
print("="*60)

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
          '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b',
          '#2980b9', '#27ae60']

# Score Distributions (compact grid)
n_datasets = len(all_genuine)
n_cols = 3
n_rows = (n_datasets + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
fig.suptitle(f'{MODEL_NAME} Score Distributions', fontsize=16, fontweight='bold')
axes_flat = axes.flatten() if n_datasets > 1 else [axes]
for i, (key, gen) in enumerate(all_genuine.items()):
    ax = axes_flat[i]
    imp = all_impostor[key]
    ax.hist(imp, bins=80, alpha=0.6, color='#e74c3c', label='Impostor', density=True)
    ax.hist(gen, bins=80, alpha=0.6, color='#2ecc71', label='Genuine', density=True)
    eer = results[key]['eer']
    thr = results[key]['eer_threshold']
    ax.axvline(thr, color='#f39c12', linestyle='--', linewidth=1.5,
               label=f'EER thr ({thr:.3f})')
    ax.set_title(f'{key}\nEER={eer*100:.2f}%', fontsize=9, fontweight='bold')
    ax.set_xlabel('Cosine Similarity', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)
# Hide unused subplots
for j in range(i+1, len(axes_flat)):
    axes_flat[j].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'score_distributions.png'), dpi=120, bbox_inches='tight')
plt.close(fig)
print("  Saved: score_distributions.png")

# DET Curves
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title(f'{MODEL_NAME} DET Curves', fontsize=16, fontweight='bold')
for i, key in enumerate(all_genuine.keys()):
    gen = all_genuine[key]; imp = all_impostor[key]
    thresholds = np.linspace(min(imp.min(), gen.min()), max(imp.max(), gen.max()), 5000)
    far = np.array([np.mean(imp >= t) for t in thresholds])
    frr = np.array([np.mean(gen < t) for t in thresholds])
    mask = (far > 0) & (frr > 0)
    if mask.sum() > 10:
        ax.plot(far[mask]*100, frr[mask]*100, color=colors[i % len(colors)],
                linewidth=2, label=f'{key} (EER={results[key]["eer"]*100:.2f}%)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('False Accept Rate (%)', fontsize=12)
ax.set_ylabel('False Reject Rate (%)', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3, which='both')
ax.plot([0.01, 100], [0.01, 100], 'k--', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'det_curves.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: det_curves.png")

# CMC Curves
cmc_keys = [k for k in results if 'cmc' in results[k]]
if cmc_keys:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(f'{MODEL_NAME} CMC Curves (Identification)', fontsize=16, fontweight='bold')
    for i, key in enumerate(cmc_keys):
        cmc_arr = np.array(results[key]['cmc'])
        max_plot = min(50, len(cmc_arr))
        ax.plot(range(1, max_plot+1), cmc_arr[:max_plot]*100, color=colors[i % len(colors)],
                linewidth=2, marker='o', markersize=3,
                label=f'{key} (R1={cmc_arr[0]*100:.1f}%)')
    ax.set_xlabel('Rank', fontsize=12); ax.set_ylabel('Identification Rate (%)', fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(1, max_plot); ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'cmc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: cmc_curves.png")

# Summary Bar Chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'{MODEL_NAME} Evaluation Summary', fontsize=16, fontweight='bold')
keys_list = list(results.keys())
x = np.arange(len(keys_list)); w = 0.6
eers = [results[k]['eer']*100 for k in keys_list]
bars = axes[0].bar(x, eers, w, color=[colors[i % len(colors)] for i in range(len(keys_list))], edgecolor='white')
axes[0].set_title('EER (%) — Lower is Better', fontsize=13, fontweight='bold')
axes[0].set_xticks(x); axes[0].set_xticklabels(keys_list, rotation=45, ha='right', fontsize=8)
axes[0].set_ylabel('EER (%)')
for bar, val in zip(bars, eers):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.2f}', ha='center', fontsize=8, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

tars = [results[k]['tar_far01'] for k in keys_list]
bars = axes[1].bar(x, tars, w, color=[colors[i % len(colors)] for i in range(len(keys_list))], edgecolor='white')
axes[1].set_title('TAR @ FAR=0.1% — Higher is Better', fontsize=13, fontweight='bold')
axes[1].set_xticks(x); axes[1].set_xticklabels(keys_list, rotation=45, ha='right', fontsize=8)
axes[1].set_ylabel('TAR (0-1)')
for bar, val in zip(bars, tars):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{val:.4f}', ha='center', fontsize=8, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

r1s = [results[k]['rank1']*100 for k in keys_list]
bars = axes[2].bar(x, r1s, w, color=[colors[i % len(colors)] for i in range(len(keys_list))], edgecolor='white')
axes[2].set_title('Rank-1 (%) — Higher is Better', fontsize=13, fontweight='bold')
axes[2].set_xticks(x); axes[2].set_xticklabels(keys_list, rotation=45, ha='right', fontsize=8)
axes[2].set_ylabel('Rank-1 (%)')
for bar, val in zip(bars, r1s):
    if val > 0:
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', fontsize=8, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'summary_bar_chart.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: summary_bar_chart.png")

# ── Performance by Fingerprint Type ──
TYPE_GROUPS = {
    'Optical':     ['FVC_FVC2002_DB1A', 'FVC_FVC2002_DB2A', 'FVC_FVC2004_DB1A',
                    'FVC_FVC2004_DB2A', 'PolyU_contact-based_fingerprints'],
    'Capacitive':  ['FVC_FVC2002_DB3A'],
    'Thermal':     ['FVC_FVC2004_DB3A'],
    'Fingerphoto': ['ISPFDv2_ISPFDv2'],
    'Contactless': ['PolyU_contactless_2d_fingerprint_images',
                    'PolyU_processed_contactless_2d_fingerprint_images'],
    'Mixed (Roll+Latent)': ['SD302_SD302'],
}
type_colors = {'Optical': '#3498db', 'Capacitive': '#2ecc71', 'Thermal': '#e74c3c',
               'Fingerphoto': '#f39c12', 'Contactless': '#9b59b6', 'Mixed (Roll+Latent)': '#34495e'}

type_eer, type_tar, type_r1 = {}, {}, {}
for tname, dsets in TYPE_GROUPS.items():
    eers_t = [results[d]['eer']*100 for d in dsets if d in results]
    tars_t = [results[d]['tar_far01'] for d in dsets if d in results]
    r1s_t  = [results[d]['rank1']*100 for d in dsets if d in results]
    if eers_t:
        type_eer[tname] = np.mean(eers_t)
        type_tar[tname] = np.mean(tars_t)
        type_r1[tname]  = np.mean(r1s_t)

if type_eer:
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(f'{MODEL_NAME} — Performance by Fingerprint Type', fontsize=16, fontweight='bold')
    types = list(type_eer.keys())
    x = np.arange(len(types))
    w = 0.55
    tc = [type_colors.get(t, '#95a5a6') for t in types]

    # EER by type
    vals = [type_eer[t] for t in types]
    bars = axes[0].bar(x, vals, w, color=tc, edgecolor='white', linewidth=1.5)
    axes[0].set_title('Avg EER (%) by Type — Lower is Better', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x); axes[0].set_xticklabels(types, rotation=30, ha='right', fontsize=9)
    axes[0].set_ylabel('EER (%)')
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # TAR by type
    vals = [type_tar[t] for t in types]
    bars = axes[1].bar(x, vals, w, color=tc, edgecolor='white', linewidth=1.5)
    axes[1].set_title('Avg TAR@FAR=0.1% by Type — Higher is Better', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x); axes[1].set_xticklabels(types, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylabel('TAR (0-1)')
    for bar, val in zip(bars, vals):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001, f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    # Rank-1 by type
    vals = [type_r1[t] for t in types]
    bars = axes[2].bar(x, vals, w, color=tc, edgecolor='white', linewidth=1.5)
    axes[2].set_title('Avg Rank-1 (%) by Type — Higher is Better', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x); axes[2].set_xticklabels(types, rotation=30, ha='right', fontsize=9)
    axes[2].set_ylabel('Rank-1 (%)')
    for bar, val in zip(bars, vals):
        axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'performance_by_type.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: performance_by_type.png")


# ═══════════════════════════════════════════════════════════════
#  11. RESULTS TABLE
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*80)
print(f"  {MODEL_NAME.upper()} EVALUATION RESULTS")
print("="*80)
header = f"{'Dataset':<40} {'EER%':>8} {'TAR@0.1%':>10} {'TAR@0.01%':>11} {'Rank-1':>8} {'Rank-5':>8}"
print(header); print("-"*80)
for key in sorted(results.keys()):
    r = results[key]
    r1 = f"{r['rank1']*100:.2f}" if r['rank1'] > 0 else "—"
    r5 = f"{r['rank5']*100:.2f}" if r['rank5'] > 0 else "—"
    print(f"{key:<40} {r['eer']*100:>7.2f}% {r['tar_far01']:>10.4f} "
          f"{r['tar_far001']:>11.4f} {r1:>8} {r5:>8}")
print("-"*80)

# LaTeX table
print(f"\n\n% LaTeX Table (copy-paste into paper)")
print("\\begin{table}[h]")
print("\\centering")
print(f"\\caption{{{MODEL_NAME} Authentication and Identification Results}}")
print(f"\\label{{tab:{MODEL_NAME.lower().replace('-','')}-results}}")
print("\\begin{tabular}{lcccc}")
print("\\toprule")
print("Dataset & EER (\\%) & TAR@FAR=0.1\\% & TAR@FAR=0.01\\% & Rank-1 \\\\")
print("\\midrule")
for key in sorted(results.keys()):
    r = results[key]
    r1 = f"{r['rank1']*100:.2f}\\%" if r['rank1'] > 0 else "---"
    print(f"{key} & {r['eer']*100:.2f} & {r['tar_far01']:.4f} "
          f"& {r['tar_far001']:.4f} & {r1} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# Save JSON
json_results = {}
for k, v in results.items():
    jr = {}
    for kk, vv in v.items():
        if kk == 'cmc': continue
        jr[kk] = float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
    json_results[k] = jr
out_name = MODEL_NAME.lower().replace('-', '') + '_results.json'
with open(os.path.join(OUTPUT_DIR, out_name), 'w') as f:
    json.dump(json_results, f, indent=2)
print(f"\nResults saved to {OUTPUT_DIR}/{out_name}")

print("\n" + "="*60)
print(f"  {MODEL_NAME.upper()} EVALUATION COMPLETE")
print("  Outputs in /kaggle/working/:")
print("    - score_distributions.png, det_curves.png")
print("    - cmc_curves.png, summary_bar_chart.png")
print(f"    - {out_name}")
print("="*60)
