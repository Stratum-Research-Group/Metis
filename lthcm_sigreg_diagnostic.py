#!/usr/bin/env python3
"""
LTHCM embedding geometry diagnostic - 2026-10-08

Save this entire txt as lthcm_sigreg_diagnostic.py. It is runnable Python.
Read-only toward Metis/LTHCM. No training, whitening of stored vectors, or repo edits.
Static-reviewed and syntax-checked, NOT executed with Torch or a real encoder here.

GROUNDING
Metis HEAD: 1fbad2f1d00e513a8a88291b934e87eeac2e6080
https://gitlab.com/akarta-315h/metis
LTHCM/lthcm/benchmarks/real/ir/scifact_eval.py:53-63,132-133:
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'), normalized encode.
dianoia/dianoia/embedding.py:20-59: TextEmbedder.encode(texts, normalize=True),
returns [N,d]. LTHCM itself consumes vectors, not one compulsory text encoder.
LTHCM/pyproject.toml:17-31: Torch/numpy core; sentence-transformers in bench extra.

LeJEPA HEAD: c293d291ca87cd4fddee9d3fffe4e914c7272052
https://github.com/galilai-group/lejepa
lejepa/multivariate/slicing.py:90-153: SlicingUnivariateTest, reduction=None,
unit Gaussian directions; seeds come from global_step, not torch.manual_seed.
lejepa/univariate/epps_pulley.py:62-99: EppsPulley(t_max=3,n_points=17),
positive-half trapezoidal characteristic-function integral, statistic multiplied by N.
The earlier sketch's -5..5 grid was not this package's actual default.
Paper: https://arxiv.org/abs/2511.08544

SETUP (isolated environment, do not replace Metis dependencies)
python3 -m venv ~/lthcm-geometry-venv
. ~/lthcm-geometry-venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy sentence-transformers
# Use a new directory, or an existing clean clone at the pinned commit:
git clone https://github.com/galilai-group/lejepa ~/lejepa-geometry
git -C ~/lejepa-geometry checkout c293d291ca87cd4fddee9d3fffe4e914c7272052
python -m pip install -e ~/lejepa-geometry
cp lthcm-sigreg-diagnostic-1fbad2f.txt lthcm_sigreg_diagnostic.py

PREFERRED: exact already-saved production vectors, N x d, float32/64 .npy, no pickle:
python lthcm_sigreg_diagnostic.py --npy corpus_embeddings.npy --out geometry-run1

OR: your real passages in JSONL, one {"text":"..."} per line. Use the same passage
chunking and encoder revision as your actual retrieval pipeline; no gold labels needed.
The default benchmark adapter directly uses the same normalized encoder calls as SciFact.
python lthcm_sigreg_diagnostic.py --texts passages.jsonl --out geometry-run1

To call Dianoia's actual TextEmbedder from your checkout instead:
python lthcm_sigreg_diagnostic.py --texts passages.jsonl --adapter dianoia \
  --metis /home/akarta/Projects/Metis --out geometry-run2
If your serving encoder differs, use --model LOCAL_MODEL_DIR or --npy. --revision
pins HF revision for benchmark adapter only; a local snapshot is preferable offline.
Encoder may download weights on the first run; diagnostic computation is CPU-only.
This bounds samples to 4096, dims to 1024, slices to 1024, four CPU threads.
Runtime depends on encoder/data/CPU; no measured runtime or GPU requirement claimed.

OUTPUT / INTERPRETATION
New output directory: embeddings.npy, report.json, slices.npz. Existing directory fails
rather than overwrites. report contains raw-centered covariance spectrum/participation
rank, standardized rank, slice stats and matched Gaussian-null ratios for three seeds,
column-shuffled control, and a retained-subspace whitened analysis control.
Participation rank = trace(C)^2 / trace(C^2); it is NOT entropy effective rank.
Constant columns are counted and excluded from standardization, not divided by zero.
Raw covariance rank sees variance anisotropy; standardized rank sees correlation.
Column shuffling preserves marginal distributions but breaks paired feature dependence.
Whitening removes measured second-order geometry and can hide exactly what we inspect.
Rank-deficient whitening uses only retained eigenvectors; report its lower dimension.
All are analysis copies. Never feed transformed vectors back into LTHCM automatically.

The earlier rule (raw participation_rank/d <0.2 AND standardized mean-stat/null >10)
is retained ONLY as an arbitrary exploratory flag, with at least 2/3 seeds agreeing.
It is NOT a p-value, calibrated alarm, proof of collapse, or permission to train.
Correlated/duplicate passages, corpus composition and unit-normalized vectors change
null expectations. Sphere-normalized retrieval vectors are not exact Gaussian vectors.
A near-null sliced result can miss a low-dimensional subspace; read the spectrum too.
Non-Gaussian/clustered/hierarchical geometry is not necessarily a retrieval defect.

NEXT ACTION
If flagged: inspect duplicates/encoder settings and actual retrieval failures first.
Only then consider a frozen-encoder small-projector A/B with an identity baseline,
held-out retrieval labels, Recall@k/nDCG, and topology/contradiction regression checks.
No SIGReg-training or LTHCM-default change is in this script. Keep query vectors separate
from corpus vectors; rerun separately if N is sufficient. Fewer than 2000 vectors is
allowed with a low-sample warning, not evidence of a training problem.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from lejepa.multivariate import SlicingUnivariateTest
from lejepa.univariate import EppsPulley


def standardized(x):
    y = x - x.mean(0)
    sd = y.square().mean(0).sqrt()
    keep = sd > 1e-8
    if not bool(keep.any()):
        raise ValueError('All columns are constant; inspect encoder/data before testing.')
    return y[:, keep] / sd[keep], int((~keep).sum())


def covariance_info(x):
    y = x.double() - x.double().mean(0)
    c = y.T @ y / (len(y) - 1)
    vals, vecs = torch.linalg.eigh(c)
    vals = vals.clamp_min(0)
    trace = vals.sum()
    pr = (trace.square() / vals.square().sum()).item() if trace > 0 else 0.0
    threshold = max(float(vals.max()) * 1e-6, 1e-12)
    keep = vals > threshold
    info = dict(participation_rank=pr, rank_fraction=pr / x.shape[1],
                numerical_rank=int(keep.sum()), dimension=x.shape[1],
                eigenvalue_threshold=threshold,
                eigenvalues_descending=vals.flip(0).tolist())
    return info, (y, vals, vecs, keep)


def slice_statistics(x, seed, slices=1024, slice_chunk=32):
    # Each chunk gets a different deterministic direction seed; same seeds across
    # controls. Bound the ECF tensor to N x slice_chunk x 17, not N x 1024 x 17.
    parts = []
    for k in range(0, slices, slice_chunk):
        test = SlicingUnivariateTest(EppsPulley(t_max=3, n_points=17),
                                     num_slices=min(slice_chunk, slices-k), reduction=None)
        test.global_step.fill_(seed * 10000 + k // slice_chunk)
        v = test(x)
        if not bool(torch.isfinite(v).all()):
            raise ValueError('Non-finite statistic; no interpretation is valid.')
        parts.append(v)
    return torch.cat(parts)


def summary(v):
    return dict(mean=float(v.mean()), median=float(v.median()),
                p95=float(torch.quantile(v, .95)), maximum=float(v.max()))


def shuffled_columns(x, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([x[torch.randperm(len(x), generator=g), j]
                        for j in range(x.shape[1])], dim=1)


def encode_texts(args):
    # Reservoir sample passages so file order does not select only an opening topic.
    rng = np.random.default_rng(args.seed)
    sample, count = [], 0
    with open(args.texts, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get('text')
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'Line {line_no}: expected nonempty text string')
            if len(text.encode('utf-8')) > 100000:
                raise ValueError(f'Line {line_no}: passage >100kB; chunk before encoding')
            count += 1
            if len(sample) < args.max_samples:
                sample.append(text)
            else:
                j = int(rng.integers(count))
                if j < args.max_samples:
                    sample[j] = text
    if len(sample) < 3:
        raise ValueError('At least three passages required')
    if args.adapter == 'dianoia':
        if args.revision:
            raise ValueError('--revision is unsupported by TextEmbedder; use a local snapshot')
        if not args.metis:
            raise ValueError('--metis checkout required for dianoia adapter')
        sys.path.insert(0, str(Path(args.metis).resolve() / 'dianoia'))
        from dianoia.embedding import TextEmbedder
        encoder = TextEmbedder(args.model, device=torch.device('cpu'))
        encoder.model.eval()
        batch_encode = lambda b: encoder.encode(b, normalize=True).cpu().float()
    else:
        from sentence_transformers import SentenceTransformer
        kwargs = dict(device='cpu')
        if args.revision:
            kwargs['revision'] = args.revision
        encoder = SentenceTransformer(args.model, **kwargs)
        encoder.eval()
        batch_encode = lambda b: encoder.encode(b, batch_size=32, normalize_embeddings=True,
                                               convert_to_tensor=True, show_progress_bar=False).cpu().float()
    chunks = [batch_encode(sample[i:i+32]) for i in range(0, len(sample), 32)]
    return torch.cat(chunks), dict(mode='texts', adapter=args.adapter, model=args.model,
                                  revision=args.revision, passages_seen=count,
                                  sampled_unique_texts=len(set(sample)))


def main():
    p = argparse.ArgumentParser(description='Read-only LTHCM geometry diagnostic')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--npy')
    src.add_argument('--texts')
    p.add_argument('--adapter', choices=['benchmark', 'dianoia'], default='benchmark')
    p.add_argument('--metis')
    p.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--revision')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--max-samples', type=int, default=4096)
    args = p.parse_args()
    if not 3 <= args.max_samples <= 4096 or not 0 <= args.seed <= 100000:
        p.error('max-samples must be 3..4096 and seed 0..100000')
    out = Path(args.out)
    if out.exists():
        p.error('Output directory exists; choose a new path')
    torch.set_num_threads(4)
    with torch.inference_mode():
        if args.npy:
            a = np.load(args.npy, mmap_mode='r', allow_pickle=False)
            if a.ndim != 2 or a.shape[0] < 3 or not 1 <= a.shape[1] <= 1024:
                raise ValueError('Expected [N>=3, 1<=d<=1024] array')
            if a.dtype.kind != 'f':
                raise ValueError('Expected float .npy, not integer/object vectors')
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(a), min(args.max_samples, len(a)), replace=False)
            x = torch.from_numpy(np.array(a[idx], dtype=np.float32, copy=True))
            provenance = dict(mode='npy', source=args.npy, rows_seen=len(a))
        else:
            x, provenance = encode_texts(args)
        x = x.cpu().float().contiguous()
        if not 1 <= x.shape[1] <= 1024 or not bool(torch.isfinite(x).all()):
            raise ValueError('Expected finite vectors with d<=1024')
        n, d = x.shape
        z, constant = standardized(x)
        raw_info, eig = covariance_info(x)
        z_info, _ = covariance_info(z)
        y, vals, vecs, keep = eig
        if not bool(keep.any()):
            raise ValueError('No covariance directions above whitening threshold; inspect scale/collapse.')
        white = ((y @ vecs[:, keep]) / vals[keep].sqrt()).float()
        arrays, runs = {}, []
        for offset in range(3):
            seed = args.seed + offset
            gen = torch.Generator().manual_seed(seed)
            null = torch.randn(n, z.shape[1], generator=gen)
            null, _ = standardized(null)  # Same estimated centering/scaling control.
            zs = slice_statistics(z, seed)
            ns = slice_statistics(null, seed)
            ss = slice_statistics(shuffled_columns(z, seed), seed)
            # Whitened controls share retained dimension, but null is also whitened.
            wn = torch.randn(n, white.shape[1], generator=gen)
            _, (wy, wvals, wvecs, wkeep) = covariance_info(wn)
            wn = ((wy @ wvecs[:, wkeep]) / wvals[wkeep].sqrt()).float()
            ws = slice_statistics(white, seed)
            wns = slice_statistics(wn, seed)
            ratio = float(zs.mean() / ns.mean().clamp_min(1e-12))
            runs.append(dict(seed=seed, standardized=summary(zs), gaussian_null=summary(ns),
                             column_shuffled=summary(ss), standardized_null_ratio=ratio,
                             whitened=summary(ws), whitened_null=summary(wns)))
            for name, stat in [('standardized', zs), ('null', ns), ('shuffled', ss),
                               ('whitened', ws), ('white_null', wns)]:
                arrays[f'{name}_seed{seed}'] = stat.numpy()
        votes = sum(r['standardized_null_ratio'] > 10 for r in runs)
        warnings = ['Not calibrated p-values; non-Gaussianity need not harm retrieval.',
                    'Whitening/standardization are diagnostic copies only.']
        if n < 2000:
            warnings.append('N<2000: low-sample diagnostic, inspect with caution.')
        if n <= d:
            warnings.append('N<=d: covariance necessarily sample-rank-limited.')
        if constant:
            warnings.append(f'{constant} constant columns excluded from standardization.')
        if white.shape[1] < d:
            warnings.append('Whitening discarded low-eigenvalue dimensions; reduced-space result.')
        report = dict(shape=[n, d], provenance=provenance, slices=1024, slice_chunk=32,
                      raw_centered_covariance=raw_info, standardized_covariance=z_info,
                      constant_columns=constant, whitened_dimension=white.shape[1], runs=runs,
                      exploratory_flag=bool(raw_info['rank_fraction'] < .2 and votes >= 2),
                      flag_rule='raw participation_rank/d <0.2 AND stat/null >10 in >=2/3 seeds; arbitrary',
                      warnings=warnings)
        out.mkdir(parents=True)
        np.save(out / 'embeddings.npy', x.numpy(), allow_pickle=False)
        np.savez_compressed(out / 'slices.npz', **arrays)
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({k: report[k] for k in ['shape', 'constant_columns',
                          'whitened_dimension', 'exploratory_flag', 'warnings']}, indent=2))
        print('raw rank fraction:', raw_info['rank_fraction'])
        print('standardized stat/null ratios:', [r['standardized_null_ratio'] for r in runs])
        print('Saved:', out / 'report.json')


if __name__ == '__main__':
    main()
