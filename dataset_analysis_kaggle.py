"""
Dataset Analysis Script — Complete Profile of All Training & Testing Datasets
══════════════════════════════════════════════════════════════════════════════

Analyzes every dataset and produces:
  - Image count, subject count
  - Resolution (min, max, avg, mode)
  - Color mode (Grayscale/RGB)
  - File formats (PNG/JPG/BMP/TIFF/WSQ)
  - File size statistics
  - Sample images grid
  - DPI estimation (from metadata if available)
  - Summary table for slides

Run on Kaggle with all datasets attached as input.
"""

import os
import sys
import glob
import json
import time
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ExifTags
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ═══════════════════════════════════════════════════════════════
#  KAGGLE INPUTS — Attach these 7 datasets to your notebook:
#
#  1. kushp3690/afr-training-real
#  2. kushpatel7391/afr-training-synthetic
#  3. kushpatel7391/afr-training-misc
#  4. abvwmc/level-three-synthetic-fingerprint-generation-l3-sf
#  5. kushp3690/molf-testing-dataset
#  6. kushp3690/polyu-testing-datasets
#  7. kushp3690/ispfdv2-testing-dataset
#
# ═══════════════════════════════════════════════════════════════

TRAINING_DATASETS = {
    # ── Dataset1: afr-training-real ──
    "SOCOFing": "/kaggle/input/datasets/kushp3690/afr-training-real/Sokoto Coventry Fingerprint Dataset/SOCOFing",
    "NIST SD300a": "/kaggle/input/datasets/kushp3690/afr-training-real/sd300a",
    "NIST SD301a": "/kaggle/input/datasets/kushp3690/afr-training-real/sd301a",
    "NIST SD301b": "/kaggle/input/datasets/kushp3690/afr-training-real/sd301b",
    "NIST SD302a (Rolled)": "/kaggle/input/datasets/kushp3690/afr-training-real/sd302a",
    "NIST SD302b (Latent)": "/kaggle/input/datasets/kushp3690/afr-training-real/sd302b",
    "NIST SD302d (Contactless)": "/kaggle/input/datasets/kushp3690/afr-training-real/sd302d",
    "NIST SD302e (Plain)": "/kaggle/input/datasets/kushp3690/afr-training-real/sd302e",
    "74034_3_En_4_MOESM1_ESM": "/kaggle/input/datasets/kushp3690/afr-training-real/74034_3_En_4_MOESM1_ESM",
    # ── Dataset2: afr-training-synthetic ──
    "AMSL SynFP P2P v1": "/kaggle/input/datasets/kushpatel7391/afr-training-synthetic/AMSL_SynFP_P2P_v1",
    "AMSL SynFP P2P v2": "/kaggle/input/datasets/kushpatel7391/afr-training-synthetic/AMSL_SynFP_P2P_v2",
    "AMSL SynFP SGR v1": "/kaggle/input/datasets/kushpatel7391/afr-training-synthetic/AMSL_SynFP_SGR_v1",
    # ── Dataset3: afr-training-misc (scan entire root) ──
    "AFR-Training-Misc (all)": "/kaggle/input/datasets/kushpatel7391/afr-training-misc",
    # ── Dataset4: validation ──
    "L3-SF v2 (Val)": "/kaggle/input/datasets/abvwmc/level-three-synthetic-fingerprint-generation-l3-sf/L3SF_V2",
}

TESTING_DATASETS = {
    # ── MOLF (actual folder names from directory structure) ──
    "MOLF DB1 Lumidgm (Optical)": "/kaggle/input/datasets/kushp3690/molf-testing-dataset/DB1_Lumidgm",
    "MOLF DB2 Secugen (Capacitive)": "/kaggle/input/datasets/kushp3690/molf-testing-dataset/DB2_Secugen",
    "MOLF DB3 CrossMatch (Contact)": "/kaggle/input/datasets/kushp3690/molf-testing-dataset/DB3_A_CrossMatchCropped",
    "MOLF DB4 Latent": "/kaggle/input/datasets/kushp3690/molf-testing-dataset/DB4_Latent",
    "MOLF DB5 SimLatent": "/kaggle/input/datasets/kushp3690/molf-testing-dataset/DB5_SimLatent",
    # ── PolyU (actual folder names from directory structure) ──
    "PolyU Contact": "/kaggle/input/datasets/kushp3690/polyu-testing-datasets/contact-based_fingerprints",
    "PolyU Contactless": "/kaggle/input/datasets/kushp3690/polyu-testing-datasets/contactless_2d_fingerprint_images",
    "PolyU Processed": "/kaggle/input/datasets/kushp3690/polyu-testing-datasets/processed_contactless_2d_fingerprint_images",
    # ── ISPFDv2 (actual folder name) ──
    "ISPFDv2": "/kaggle/input/datasets/kushp3690/ispfdv2-testing-dataset/ISPFDv2",
    # ── NIST SD302 test split ──
    "SD302 (Test Split)": "/kaggle/input/datasets/kushp3690/afr-training-real/sd302a",
}

SAVE_DIR = "/kaggle/working"
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.wsq', '.pgm', '.ppm', '.gif'}


# ═══════════════════════════════════════════════════════════════
#  ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def find_images(root_path):
    """Find ALL image files under a directory (no limit)."""
    images = []
    if not os.path.exists(root_path):
        return images
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                images.append(os.path.join(dirpath, fname))
    return images


def estimate_subjects(image_paths, root_path):
    """Estimate number of subjects from directory structure or filenames."""
    subjects = set()
    for p in image_paths:
        rel = os.path.relpath(p, root_path)
        parts = rel.replace('\\', '/').split('/')
        if len(parts) >= 2:
            # Subject is usually the parent folder or prefix
            subjects.add(parts[-2] if len(parts) > 1 else parts[0])
        else:
            # Try filename prefix
            fname = os.path.splitext(os.path.basename(p))[0]
            # Common patterns: 101_1.tif, S0001_L_1.png
            if '_' in fname:
                subjects.add(fname.rsplit('_', 1)[0])
            else:
                subjects.add(fname[:4])
    return len(subjects)


def analyze_dataset(name, root_path, sample_size=500):
    """Analyze a single dataset and return comprehensive stats."""
    print(f"\n  📂 Analyzing: {name}")
    
    result = {
        'name': name,
        'path': root_path,
        'exists': os.path.exists(root_path),
        'total_images': 0,
        'estimated_subjects': 0,
        'widths': [],
        'heights': [],
        'color_modes': Counter(),
        'channels': Counter(),
        'file_formats': Counter(),
        'file_sizes_kb': [],
        'bit_depths': Counter(),
        'dpi_values': [],
        'sample_images': [],
        'errors': 0,
    }
    
    if not result['exists']:
        print(f"     ❌ Path not found: {root_path}")
        return result
    
    all_images = find_images(root_path)
    result['total_images'] = len(all_images)
    result['estimated_subjects'] = estimate_subjects(all_images, root_path)
    
    if not all_images:
        print(f"     ⚠️ No images found")
        return result
    
    # Sample images for detailed analysis
    sample_indices = np.random.RandomState(42).choice(
        len(all_images), min(sample_size, len(all_images)), replace=False
    )
    sampled = [all_images[i] for i in sample_indices]
    
    for fpath in sampled:
        try:
            # File size
            fsize_kb = os.path.getsize(fpath) / 1024
            result['file_sizes_kb'].append(fsize_kb)
            
            # File format
            ext = os.path.splitext(fpath)[1].lower()
            result['file_formats'][ext] += 1
            
            # Open image
            img = Image.open(fpath)
            w, h = img.size
            result['widths'].append(w)
            result['heights'].append(h)
            result['color_modes'][img.mode] += 1
            
            # Channels
            if img.mode == 'L':
                result['channels']['Grayscale (1ch)'] += 1
            elif img.mode == 'RGB':
                result['channels']['RGB (3ch)'] += 1
            elif img.mode == 'RGBA':
                result['channels']['RGBA (4ch)'] += 1
            elif img.mode == 'P':
                result['channels']['Palette (1ch)'] += 1
            else:
                result['channels'][f'{img.mode}'] += 1
            
            # Bit depth
            if hasattr(img, 'bits'):
                result['bit_depths'][img.bits] += 1
            elif img.mode == 'L':
                result['bit_depths']['8-bit'] += 1
            elif img.mode in ('RGB', 'RGBA'):
                result['bit_depths']['24-bit'] += 1
            elif img.mode == 'I;16':
                result['bit_depths']['16-bit'] += 1
            
            # DPI from EXIF/metadata
            try:
                dpi = img.info.get('dpi', None)
                if dpi and isinstance(dpi, tuple) and dpi[0] > 0:
                    result['dpi_values'].append(int(dpi[0]))
            except:
                pass
            
            # Keep a few samples for visualization
            if len(result['sample_images']) < 6:
                result['sample_images'].append(fpath)
            
            img.close()
        except Exception as e:
            result['errors'] += 1
    
    # Print summary
    imgs_per_subj = result['total_images'] / max(result['estimated_subjects'], 1)
    print(f"     Images: {result['total_images']:,} | Subjects: ~{result['estimated_subjects']:,} | ~{imgs_per_subj:.1f} imgs/subject")
    
    if result['widths']:
        w_mode = Counter(result['widths']).most_common(1)[0]
        h_mode = Counter(result['heights']).most_common(1)[0]
        print(f"     Resolution: {w_mode[0]}×{h_mode[0]} (most common, {w_mode[1]}/{len(result['widths'])} samples)")
        print(f"     Range: {min(result['widths'])}×{min(result['heights'])} → {max(result['widths'])}×{max(result['heights'])}")
    
    color_str = ', '.join(f"{m}: {c}" for m, c in result['color_modes'].most_common(3))
    print(f"     Color: {color_str}")
    
    fmt_str = ', '.join(f"{f}: {c}" for f, c in result['file_formats'].most_common(3))
    print(f"     Formats: {fmt_str}")
    
    if result['file_sizes_kb']:
        print(f"     File size: {np.mean(result['file_sizes_kb']):.1f} KB avg ({min(result['file_sizes_kb']):.1f}–{max(result['file_sizes_kb']):.1f} KB)")
    
    if result['dpi_values']:
        print(f"     DPI: {Counter(result['dpi_values']).most_common(1)[0][0]} (from metadata)")
    
    return result


# ═══════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def create_sample_grid(all_results, save_path, title="Dataset Sample Gallery"):
    """Create a grid showing sample images from each dataset."""
    datasets_with_samples = [r for r in all_results if r['sample_images']]
    if not datasets_with_samples:
        return
    
    n = len(datasets_with_samples)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]
    
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    
    for idx, result in enumerate(datasets_with_samples):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        
        if result['sample_images']:
            try:
                img = Image.open(result['sample_images'][0]).convert('L')
                ax.imshow(np.array(img), cmap='gray')
            except:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
        
        # Title with key info
        w_mode = Counter(result['widths']).most_common(1)[0][0] if result['widths'] else '?'
        h_mode = Counter(result['heights']).most_common(1)[0][0] if result['heights'] else '?'
        color = result['color_modes'].most_common(1)[0][0] if result['color_modes'] else '?'
        
        ax.set_title(f"{result['name']}\n{w_mode}×{h_mode} | {color} | {result['total_images']:,} imgs",
                     fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Hide empty cells
    for idx in range(len(datasets_with_samples), rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  💾 Saved: {save_path}")


def create_summary_table(all_results, save_path, title="Dataset Summary"):
    """Create a formatted table image summarizing all datasets."""
    fig, ax = plt.subplots(figsize=(18, max(4, len(all_results) * 0.5 + 2)))
    ax.axis('off')
    
    # Table data
    headers = ['Dataset', 'Images', 'Subjects', 'Imgs/Subj', 'Resolution (Mode)', 'Color', 'Format', 'Avg Size', 'DPI']
    table_data = []
    
    for r in all_results:
        if not r['exists'] or r['total_images'] == 0:
            table_data.append([r['name'], '—', '—', '—', '—', '—', '—', '—', '—'])
            continue
        
        imgs = f"{r['total_images']:,}"
        subjs = f"~{r['estimated_subjects']:,}"
        ips = f"{r['total_images']/max(r['estimated_subjects'],1):.1f}"
        
        if r['widths']:
            w_mode = Counter(r['widths']).most_common(1)[0][0]
            h_mode = Counter(r['heights']).most_common(1)[0][0]
            res = f"{w_mode}×{h_mode}"
        else:
            res = '—'
        
        color = r['color_modes'].most_common(1)[0][0] if r['color_modes'] else '—'
        fmt = r['file_formats'].most_common(1)[0][0] if r['file_formats'] else '—'
        avg_size = f"{np.mean(r['file_sizes_kb']):.0f} KB" if r['file_sizes_kb'] else '—'
        dpi = str(Counter(r['dpi_values']).most_common(1)[0][0]) if r['dpi_values'] else '—'
        
        table_data.append([r['name'], imgs, subjs, ips, res, color, fmt, avg_size, dpi])
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    
    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor('#2C3E50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        color = '#ECF0F1' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[i, j].set_facecolor(color)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  💾 Saved: {save_path}")


def create_resolution_chart(all_results, save_path):
    """Bar chart of dataset resolutions."""
    names = []
    areas = []
    colors = []
    
    for r in all_results:
        if r['widths']:
            w = Counter(r['widths']).most_common(1)[0][0]
            h = Counter(r['heights']).most_common(1)[0][0]
            names.append(r['name'])
            areas.append(w * h)
            colors.append('#3498DB' if 'FVC' in r['name'] or 'PolyU' in r['name'] else '#E67E22')
    
    if not names:
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(range(len(names)), areas, color=colors, edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Resolution (Width × Height pixels)', fontsize=12)
    ax.set_title('Dataset Resolution Comparison\n(All resized to 224×224 for training)', fontsize=14, fontweight='bold')
    
    # Add resolution labels
    for i, (bar, name) in enumerate(zip(bars, names)):
        r = all_results[i] if i < len(all_results) else None
        if r and r['widths']:
            w = Counter(r['widths']).most_common(1)[0][0]
            h = Counter(r['heights']).most_common(1)[0][0]
            ax.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2,
                   f'{w}×{h}', va='center', fontsize=8)
    
    # Add target line
    ax.axvline(x=224*224, color='red', linestyle='--', alpha=0.7, label='Training input (224×224)')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  💾 Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("=" * 70)
    print("  COMPREHENSIVE DATASET ANALYSIS")
    print("=" * 70)
    
    # ── Analyze Training Datasets ──
    print("\n\n" + "=" * 70)
    print("  TRAINING DATASETS")
    print("=" * 70)
    
    train_results = []
    for name, path in TRAINING_DATASETS.items():
        r = analyze_dataset(name, path)
        train_results.append(r)
    
    # ── Analyze Testing Datasets ──
    print("\n\n" + "=" * 70)
    print("  TESTING DATASETS")
    print("=" * 70)
    
    test_results = []
    for name, path in TESTING_DATASETS.items():
        r = analyze_dataset(name, path)
        test_results.append(r)
    
    # ── Create Visualizations ──
    print("\n\n" + "=" * 70)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    create_summary_table(train_results,
                        os.path.join(SAVE_DIR, "training_dataset_summary.png"),
                        "Training Dataset Summary")
    
    create_summary_table(test_results,
                        os.path.join(SAVE_DIR, "testing_dataset_summary.png"),
                        "Testing Dataset Summary")
    
    create_sample_grid(train_results,
                      os.path.join(SAVE_DIR, "training_dataset_samples.png"),
                      "Training Dataset Sample Gallery")
    
    create_sample_grid(test_results,
                      os.path.join(SAVE_DIR, "testing_dataset_samples.png"),
                      "Testing Dataset Sample Gallery")
    
    all_results = train_results + test_results
    create_resolution_chart([r for r in all_results if r['widths']],
                           os.path.join(SAVE_DIR, "resolution_comparison.png"))
    
    # ── Print Grand Summary ──
    print("\n\n" + "=" * 70)
    print("  GRAND SUMMARY")
    print("=" * 70)
    
    total_train = sum(r['total_images'] for r in train_results)
    total_test = sum(r['total_images'] for r in test_results)
    total_train_subj = sum(r['estimated_subjects'] for r in train_results)
    total_test_subj = sum(r['estimated_subjects'] for r in test_results)
    
    print(f"\n  Training:")
    print(f"    Total images:   {total_train:,}")
    print(f"    Total subjects: ~{total_train_subj:,}")
    print(f"    Datasets:       {len(train_results)}")
    
    print(f"\n  Testing:")
    print(f"    Total images:   {total_test:,}")
    print(f"    Total subjects: ~{total_test_subj:,}")
    print(f"    Datasets:       {len(test_results)}")
    
    # Resolution summary
    print(f"\n  Resolution Diversity:")
    for r in all_results:
        if r['widths']:
            w = Counter(r['widths']).most_common(1)[0][0]
            h = Counter(r['heights']).most_common(1)[0][0]
            color = r['color_modes'].most_common(1)[0][0] if r['color_modes'] else '?'
            print(f"    {r['name']:<30s} → {w:>5d}×{h:<5d} ({color})")
    
    print(f"\n  All resized to 224×224×3 (RGB) for model input")
    
    # Save JSON report
    report = {
        'training': [{
            'name': r['name'],
            'images': r['total_images'],
            'subjects': r['estimated_subjects'],
            'resolution_mode': f"{Counter(r['widths']).most_common(1)[0][0]}x{Counter(r['heights']).most_common(1)[0][0]}" if r['widths'] else 'N/A',
            'color': r['color_modes'].most_common(1)[0][0] if r['color_modes'] else 'N/A',
            'format': r['file_formats'].most_common(1)[0][0] if r['file_formats'] else 'N/A',
            'avg_size_kb': round(np.mean(r['file_sizes_kb']), 1) if r['file_sizes_kb'] else 0,
            'dpi': Counter(r['dpi_values']).most_common(1)[0][0] if r['dpi_values'] else 'Unknown',
        } for r in train_results if r['exists']],
        'testing': [{
            'name': r['name'],
            'images': r['total_images'],
            'subjects': r['estimated_subjects'],
            'resolution_mode': f"{Counter(r['widths']).most_common(1)[0][0]}x{Counter(r['heights']).most_common(1)[0][0]}" if r['widths'] else 'N/A',
            'color': r['color_modes'].most_common(1)[0][0] if r['color_modes'] else 'N/A',
            'format': r['file_formats'].most_common(1)[0][0] if r['file_formats'] else 'N/A',
            'avg_size_kb': round(np.mean(r['file_sizes_kb']), 1) if r['file_sizes_kb'] else 0,
            'dpi': Counter(r['dpi_values']).most_common(1)[0][0] if r['dpi_values'] else 'Unknown',
        } for r in test_results if r['exists']],
    }
    
    json_path = os.path.join(SAVE_DIR, "dataset_analysis_report.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  💾 JSON report: {json_path}")
    
    print("\n" + "=" * 70)
    print("  ✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n  Output files in {SAVE_DIR}:")
    print(f"    • training_dataset_summary.png   (table for slides)")
    print(f"    • testing_dataset_summary.png    (table for slides)")
    print(f"    • training_dataset_samples.png   (sample gallery)")
    print(f"    • testing_dataset_samples.png    (sample gallery)")
    print(f"    • resolution_comparison.png      (resolution bar chart)")
    print(f"    • dataset_analysis_report.json   (full report)")
