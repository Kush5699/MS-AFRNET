"""
MS-AFR-Net Evaluation — Part 1 of 2
Extracts embeddings from all test datasets using the trained checkpoint.
Saves embeddings + metadata to /kaggle/working/ for Part 2 (metrics).

Kaggle Inputs Required:
  Model:  talakshrabari/ms-afrnet-best-version2  (checkpoint)
  Data1:  kushp3690/molf-testing-dataset
  Data2:  kushp3690/polyu-testing-datasets
  Data3:  kushp3690/ispfdv2-testing-dataset
  Data4:  kushp3690/afr-training-real      (FVC + SD302 test splits)
"""

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
    """Extract 768-d MS-AFR-Net embeddings (z_c + z_ms)."""
    ds = EvalDataset(file_list)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)
    all_emb = []
    for imgs, _ in dl:
        imgs = imgs.to(DEVICE)
        with autocast(dtype=torch.float16):
            emb = model(imgs)
        emb = F.normalize(emb.float(), p=2, dim=1)
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
            for r, dirs, _ in os.walk(base):
                for d in dirs:
                    if d.lower() == db_name.lower():
                        db_path = os.path.join(r, d)
                        break
                if db_path: break
        if db_path:
            imgs = scan_images(db_path)
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
        print(f"\n  {group_name}/{subset_name}: {len(imgs)} images...")
        t0 = time.time()
        emb = extract_embeddings(model, imgs)
        elapsed = time.time() - t0
        print(f"    Extracted {emb.shape} in {elapsed:.1f}s ({len(imgs)/elapsed:.0f} img/s)")

        # Save embeddings
        safe_name = f"{group_name}_{subset_name}".replace('/', '_')
        np.save(os.path.join(OUTPUT_DIR, f"emb_{safe_name}.npy"), emb)

        # Save file list + subject mapping
        file_subjects = []
        subjects_dict = subset_data['subjects']
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
print("  PART 1 COMPLETE")
print("  All embeddings saved to /kaggle/working/")
print("  Run Part 2 to compute EER, Rank-1, and generate plots.")
print("="*60)
