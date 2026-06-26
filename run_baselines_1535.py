#!/usr/bin/env python
"""
Shard the 3 zero-shot point-cloud baselines (PartSLIP, Find3D, PatchAlign3D)
over the 1535-entity corrected test set across all GPUs.

Each baseline runs in its OWN conda env (the model forward needs that env);
GT-feature extraction is shelled to LISA_multi_view internally by each script.

The 1535 entities are exposed deterministically by CAD_ViewRank_Dataset when
constructed with:
    dataset_log_path = val_dataset_1535.log         (218 folders)
    entity_allowlist = val_dataset_1535_entities.txt (1535 keys)
We shard by dataset index range [start_idx, end_idx) so each process handles a
disjoint slice. Each (config, shard) writes its own experiment_name dir, so the
per-shard localization_metrics_*.log files never collide. Aggregate afterwards
with aggregate_baselines_1535.py.

A lightweight supervisor keeps at most --per-gpu jobs on each GPU, assigning the
next pending job to the least-loaded eligible GPU and refilling as jobs finish.
"""
import os
import sys
import time
import argparse
import subprocess

# Repo root = directory holding this script (works wherever the repo is cloned).
ROOT = os.path.dirname(os.path.abspath(__file__))
VAL_LOG = os.path.join(ROOT, "configs", "val_dataset_1535.log")
ALLOWLIST = os.path.join(ROOT, "configs", "val_dataset_1535_entities.txt")
LOG_DIR = os.path.join(ROOT, "run_logs_baselines")

# Each baseline runs in its OWN conda env (incompatible deps). Point these at
# the python interpreter of the matching env, or override via environment vars.
PARTSLIP_PY = os.environ.get("PARTSLIP_PY", "python")
FIND3D_PY = os.environ.get("FIND3D_PY", "python")
PATCHALIGN_PY = os.environ.get("PATCHALIGN_PY", "python")

# Vendored baseline repos live under baselines/<name>/ (see baselines/README.md).
PARTSLIP_SCRIPT = os.path.join(ROOT, "baselines", "partslip", "partslip_geolocalization.py")
FIND3D_SCRIPT = os.path.join(ROOT, "baselines", "find3d", "find3d_geolocalization.py")
PATCHALIGN_SCRIPT = os.path.join(ROOT, "baselines", "patchalign3d", "patchalign3d_geolocalization.py")

# (key, python, script, extra_args, experiment_base)
# "default" == paper-faithful variant (table "Default"); "toppcd" == Top-PCD variant.
CONFIGS = {
    "partslip_default":   (PARTSLIP_PY, PARTSLIP_SCRIPT,
                           ["--preset", "conf"], "GeLoM_PartSLIP_default"),
    "partslip_toppcd":    (PARTSLIP_PY, PARTSLIP_SCRIPT,
                           ["--preset", "topk_pct"], "GeLoM_PartSLIP_toppcd"),
    "find3d_default":     (FIND3D_PY, FIND3D_SCRIPT,
                           ["--n_points", "5000", "--preset", "default"], "GeLoM_Find3D_default"),
    "find3d_toppcd":      (FIND3D_PY, FIND3D_SCRIPT,
                           ["--n_points", "5000", "--preset", "topk"], "GeLoM_Find3D_toppcd"),
    "patchalign_default": (PATCHALIGN_PY, PATCHALIGN_SCRIPT,
                           ["--preset", "default"], "GeLoM_PatchAlign3D_default"),
    "patchalign_toppcd":  (PATCHALIGN_PY, PATCHALIGN_SCRIPT,
                           ["--preset", "topk"], "GeLoM_PatchAlign3D_toppcd"),
}


def count_entities():
    n = 0
    with open(ALLOWLIST) as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                n += 1
    return n


def shard_ranges(n_total, n_shards):
    """Contiguous, balanced [start, end) index ranges."""
    base = n_total // n_shards
    rem = n_total % n_shards
    ranges, start = [], 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        if size == 0:
            continue
        ranges.append((start, start + size))
        start += size
    return ranges


def build_jobs(config_keys, n_shards, n_total):
    jobs = []
    for key in config_keys:
        py, script, extra, exp_base = CONFIGS[key]
        for sidx, (s, e) in enumerate(shard_ranges(n_total, n_shards)):
            exp_name = f"{exp_base}_1535_shard{sidx:02d}"
            cmd = [
                py, "-u", script,
                "--val_dataset_log", VAL_LOG,
                "--entity_allowlist", ALLOWLIST,
                "--start_idx", str(s),
                "--end_idx", str(e),
                "--experiment_name", exp_name,
                *extra,
            ]
            jobs.append({
                "key": key, "shard": sidx, "cmd": cmd,
                "log": os.path.join(LOG_DIR, f"{key}__shard{sidx:02d}.log"),
            })
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS.keys()),
                    choices=list(CONFIGS.keys()),
                    help="Which baseline configs to run (default: all 6).")
    ap.add_argument("--shards", type=int, default=6,
                    help="Number of index-shards per config.")
    ap.add_argument("--gpus", type=str, default="0,1,2,3",
                    help="Comma-separated GPU ids to use.")
    ap.add_argument("--per-gpu", type=int, default=3,
                    help="Max concurrent jobs per GPU.")
    ap.add_argument("--stagger", type=float, default=8.0,
                    help="Seconds to wait between consecutive launches.")
    ap.add_argument("--poll", type=float, default=10.0,
                    help="Supervisor poll interval (s).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    os.makedirs(LOG_DIR, exist_ok=True)
    n_total = count_entities()
    jobs = build_jobs(args.configs, args.shards, n_total)

    print(f"[plan] {len(jobs)} jobs ({len(args.configs)} configs x "
          f"{args.shards} shards), {n_total} entities, gpus={gpus}, "
          f"per_gpu={args.per_gpu} -> max {len(gpus) * args.per_gpu} concurrent")
    for j in jobs:
        print(f"  {j['key']:20s} shard{j['shard']:02d}: {' '.join(j['cmd'][3:])}")
    if args.dry_run:
        return

    pending = list(jobs)
    running = []          # list of dicts: {job, proc, gpu, lf}
    gpu_load = {g: 0 for g in gpus}
    done, failed = [], []

    def launch(job, gpu):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        lf = open(job["log"], "w")
        lf.write(f"# CUDA_VISIBLE_DEVICES={gpu}\n# {' '.join(job['cmd'])}\n")
        lf.flush()
        proc = subprocess.Popen(job["cmd"], stdout=lf, stderr=subprocess.STDOUT,
                                env=env, cwd=ROOT)
        print(f"[launch] {job['key']} shard{job['shard']:02d} -> gpu{gpu} "
              f"(pid {proc.pid}) log={os.path.relpath(job['log'], ROOT)}")
        return {"job": job, "proc": proc, "gpu": gpu, "lf": lf}

    while pending or running:
        # Fill free slots, least-loaded GPU first.
        progressed = True
        while pending and progressed:
            progressed = False
            elig = [g for g in gpus if gpu_load[g] < args.per_gpu]
            if not elig:
                break
            gpu = min(elig, key=lambda g: gpu_load[g])
            job = pending.pop(0)
            r = launch(job, gpu)
            running.append(r)
            gpu_load[gpu] += 1
            progressed = True
            time.sleep(args.stagger)

        time.sleep(args.poll)

        still = []
        for r in running:
            rc = r["proc"].poll()
            if rc is None:
                still.append(r)
                continue
            r["lf"].close()
            gpu_load[r["gpu"]] -= 1
            tag = f"{r['job']['key']} shard{r['job']['shard']:02d}"
            if rc == 0:
                done.append(r["job"])
                print(f"[done] {tag} (rc=0)  [{len(done)} done / "
                      f"{len(failed)} failed / {len(still)+len(pending)} left]")
            else:
                failed.append((r["job"], rc))
                print(f"[FAIL] {tag} rc={rc} -- see {os.path.relpath(r['job']['log'], ROOT)}")
        running = still

    print(f"\n[finished] {len(done)} ok, {len(failed)} failed")
    for job, rc in failed:
        print(f"  FAILED {job['key']} shard{job['shard']:02d} rc={rc}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
