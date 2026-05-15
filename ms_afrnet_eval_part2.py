"""
MS-AFR-Net Evaluation — Part 2 of 2
Loads embeddings from Part 1 and computes:
  - EER (Equal Error Rate) for authentication
  - TAR @ FAR=0.1% and FAR=0.01%
  - Rank-1 / Rank-5 identification accuracy
  - DET curves, CMC curves, score distributions
  - Final comparison table (LaTeX-ready)

Prerequisite: Run Part 1 first. Embeddings must be in /kaggle/working/.
"""

import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from collections import defaultdict

warnings.filterwarnings('ignore')

MODEL_NAME = "MS-AFR-Net"

# ── UPDATE THIS PATH after uploading Part 1's output as a Kaggle dataset ──
INPUT_DIR  = "/kaggle/input/datasets/talakshrabari/embeddings-v2"
OUTPUT_DIR = "/kaggle/working"


# ═══════════════════════════════════════════════════════════════
#  1. LOAD EMBEDDINGS + METADATA
# ═══════════════════════════════════════════════════════════════

def load_all_embeddings(input_dir=INPUT_DIR):
    """Load all emb_*.npy + meta_*.json pairs from Part 1's uploaded output."""
    data = {}
    for f in sorted(os.listdir(input_dir)):
        if f.startswith("emb_") and f.endswith(".npy"):
            key = f[4:-4]  # strip "emb_" and ".npy"
            emb = np.load(os.path.join(input_dir, f))
            meta_file = os.path.join(input_dir, f"meta_{key}.json")
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as mf:
                    meta = json.load(mf)
            else:
                meta = [{'filepath': f'img_{i}', 'subject_id': str(i)} for i in range(len(emb))]
            data[key] = {'embeddings': emb, 'meta': meta}
            print(f"  Loaded {key}: {emb.shape}")
    return data

print("Loading embeddings from Part 1...")
all_data = load_all_embeddings()

if not all_data:
    print("ERROR: No embeddings found. Run Part 1 first!")
    import sys; sys.exit(1)


# ═══════════════════════════════════════════════════════════════
#  2. BIOMETRIC METRIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════

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
    eer_threshold = thresholds[idx]
    return eer, eer_threshold, thresholds, far, frr


def compute_tar_at_far(genuine, impostor, target_far=0.001):
    thresholds = np.linspace(impostor.min(), impostor.max(), 10000)
    for t in reversed(thresholds):
        far = np.mean(impostor >= t)
        if far <= target_far:
            tar = np.mean(genuine >= t)
            return tar, far, t
    return 0.0, target_far, thresholds[-1]


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
#  3. EVALUATE EACH DATASET
# ═══════════════════════════════════════════════════════════════

results = {}
all_genuine = {}
all_impostor = {}

print("\n" + "="*60)
print("  COMPUTING BIOMETRIC METRICS")
print("="*60)

for key, dset in all_data.items():
    emb = dset['embeddings']
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

    print(f"\n  {key}: {len(emb)} images, {len(unique_subs)} subjects "
          f"({len(multi_sample_subs)} with 2+ samples)")

    genuine, impostor = generate_genuine_impostor_scores(emb, sids)
    print(f"    Genuine pairs: {len(genuine):,}, Impostor pairs: {len(impostor):,}")

    if len(genuine) < 5 or len(impostor) < 5:
        print(f"    Too few pairs, skipping")
        continue

    eer, eer_thresh, thresholds, far_curve, frr_curve = compute_eer(genuine, impostor)
    tar_01, far_01_actual, _ = compute_tar_at_far(genuine, impostor, 0.001)
    tar_001, far_001_actual, _ = compute_tar_at_far(genuine, impostor, 0.0001)

    print(f"    EER:            {eer*100:.2f}%")
    print(f"    TAR@FAR=0.1%:   {tar_01*100:.2f}%")
    print(f"    TAR@FAR=0.01%:  {tar_001*100:.2f}%")

    # CMC
    gallery_idx, probe_idx = [], []
    seen = set()
    for i, s in enumerate(sids):
        if s not in seen and s in multi_sample_subs:
            gallery_idx.append(i)
            seen.add(s)
        elif s in multi_sample_subs:
            probe_idx.append(i)

    rank1, rank5 = 0.0, 0.0
    cmc = None
    if len(gallery_idx) >= 2 and len(probe_idx) >= 2:
        g_emb = emb[gallery_idx]
        g_ids = [sids[i] for i in gallery_idx]
        p_emb = emb[probe_idx]
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
        'n_images': len(emb), 'n_subjects': len(unique_subs),
        'n_genuine': len(genuine), 'n_impostor': len(impostor),
    }
    all_genuine[key] = genuine
    all_impostor[key] = impostor
    if cmc is not None:
        results[key]['cmc'] = cmc.tolist()


# ═══════════════════════════════════════════════════════════════
#  4. VISUALIZATION
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  GENERATING PLOTS")
print("="*60)

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
          '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b',
          '#2980b9', '#27ae60']

# ── 4.1: Score Distribution plots (compact grid) ──
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
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)

# Hide unused subplots
for j in range(i+1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'score_distributions.png'), dpi=120, bbox_inches='tight')
plt.close(fig)
print("  Saved: score_distributions.png")

# ── 4.2: DET Curves ──
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title(f'{MODEL_NAME} DET Curves', fontsize=16, fontweight='bold')

for i, key in enumerate(all_genuine.keys()):
    gen = all_genuine[key]
    imp = all_impostor[key]
    thresholds = np.linspace(min(imp.min(), gen.min()), max(imp.max(), gen.max()), 5000)
    far = np.array([np.mean(imp >= t) for t in thresholds])
    frr = np.array([np.mean(gen < t) for t in thresholds])
    mask = (far > 0) & (frr > 0)
    if mask.sum() > 10:
        ax.plot(far[mask]*100, frr[mask]*100, color=colors[i % len(colors)],
                linewidth=2, label=f'{key} (EER={results[key]["eer"]*100:.2f}%)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('False Accept Rate (%)', fontsize=12)
ax.set_ylabel('False Reject Rate (%)', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3, which='both')
ax.plot([0.01, 100], [0.01, 100], 'k--', alpha=0.3, label='EER line')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'det_curves.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: det_curves.png")

# ── 4.3: CMC Curves ──
cmc_keys = [k for k in results if 'cmc' in results[k]]
if cmc_keys:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(f'{MODEL_NAME} CMC Curves (Identification)', fontsize=16, fontweight='bold')
    for i, key in enumerate(cmc_keys):
        cmc = np.array(results[key]['cmc'])
        max_plot = min(50, len(cmc))
        ax.plot(range(1, max_plot+1), cmc[:max_plot]*100, color=colors[i % len(colors)],
                linewidth=2, marker='o', markersize=3,
                label=f'{key} (R1={cmc[0]*100:.1f}%)')
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Identification Rate (%)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(1, max_plot)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'cmc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: cmc_curves.png")

# ── 4.4: Summary Bar Chart ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'{MODEL_NAME} Evaluation Summary', fontsize=16, fontweight='bold')

keys = list(results.keys())
x = np.arange(len(keys))
w = 0.6

# EER
eers = [results[k]['eer']*100 for k in keys]
bars = axes[0].bar(x, eers, w, color=[colors[i % len(colors)] for i in range(len(keys))], edgecolor='white')
axes[0].set_title('EER (%) — Lower is Better', fontsize=13, fontweight='bold')
axes[0].set_xticks(x); axes[0].set_xticklabels(keys, rotation=45, ha='right', fontsize=8)
axes[0].set_ylabel('EER (%)')
for bar, val in zip(bars, eers):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.2f}', ha='center', fontsize=8, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# TAR@FAR=0.1%
tars = [results[k]['tar_far01']*100 for k in keys]
bars = axes[1].bar(x, tars, w, color=[colors[i % len(colors)] for i in range(len(keys))], edgecolor='white')
axes[1].set_title('TAR @ FAR=0.1% — Higher is Better', fontsize=13, fontweight='bold')
axes[1].set_xticks(x); axes[1].set_xticklabels(keys, rotation=45, ha='right', fontsize=8)
axes[1].set_ylabel('TAR (%)')
for bar, val in zip(bars, tars):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', fontsize=8, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Rank-1
r1s = [results[k]['rank1']*100 for k in keys]
bars = axes[2].bar(x, r1s, w, color=[colors[i % len(colors)] for i in range(len(keys))], edgecolor='white')
axes[2].set_title('Rank-1 (%) — Higher is Better', fontsize=13, fontweight='bold')
axes[2].set_xticks(x); axes[2].set_xticklabels(keys, rotation=45, ha='right', fontsize=8)
axes[2].set_ylabel('Rank-1 (%)')
for bar, val in zip(bars, r1s):
    if val > 0:
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', fontsize=8, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'summary_bar_chart.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: summary_bar_chart.png")


# ═══════════════════════════════════════════════════════════════
#  5. RESULTS TABLE
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*80)
print(f"  {MODEL_NAME.upper()} EVALUATION RESULTS")
print("="*80)

header = f"{'Dataset':<40} {'EER%':>8} {'TAR@0.1%':>10} {'TAR@0.01%':>11} {'Rank-1':>8} {'Rank-5':>8}"
print(header)
print("-"*80)

for key in sorted(results.keys()):
    r = results[key]
    r1 = f"{r['rank1']*100:.2f}" if r['rank1'] > 0 else "—"
    r5 = f"{r['rank5']*100:.2f}" if r['rank5'] > 0 else "—"
    print(f"{key:<40} {r['eer']*100:>7.2f}% {r['tar_far01']:>10.4f} "
          f"{r['tar_far001']:>11.4f} {r1:>8} {r5:>8}")

print("-"*80)

# ── LaTeX table ──
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

# ── Save JSON results ──
json_results = {}
for k, v in results.items():
    jr = {}
    for kk, vv in v.items():
        if kk == 'cmc':
            continue
        jr[kk] = float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
    json_results[k] = jr
with open(os.path.join(OUTPUT_DIR, 'msafrnet_results.json'), 'w') as f:
    json.dump(json_results, f, indent=2)
print(f"\nResults saved to {OUTPUT_DIR}/msafrnet_results.json")

print("\n" + "="*60)
print(f"  {MODEL_NAME.upper()} EVALUATION COMPLETE")
print("  Outputs in /kaggle/working/:")
print("    - score_distributions.png, det_curves.png")
print("    - cmc_curves.png, summary_bar_chart.png")
print("    - msafrnet_results.json")
print("="*60)
