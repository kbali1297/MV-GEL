#!/usr/bin/env python
"""Launch all 12 GeoLocLM_exp.py evaluation runs across 4 H100 GPUs.

Grid (2 Seg. VLMs x 6 view selectors = 12 runs):
    Seg. VLM      : LISA-CAD, LISA-Vanilla
    View Selector : cross_attention, film, no_fusion, only_clip, random, GT

Scheduling:
    - 12 runs / 4 GPUs = 3 runs per GPU (round-robin -> uniform load).
    - Each run ~17 GB -> ~51 GB/GPU (H100 80 GB: comfortably fits).
    - GeoLocLM_exp.py hardcodes device="cuda:0", so every process is pinned to
      one physical GPU via CUDA_VISIBLE_DEVICES (which it then sees as cuda:0).

Logs:
    One file per run at run_logs_GeoLocLM/<name>.gpu<N>.log (live, line-buffered).
    A summary table is printed at the end with exit codes and log paths.

Usage:
    python run_GeoLocLM_all.py                 # launch everything
    python run_GeoLocLM_all.py --dry-run       # print the plan, launch nothing
    python run_GeoLocLM_all.py --num-gpus 4 --num-top-views 5 --view-nms 0

    For specific entities:
    nohup /data/1bali/miniforge3/envs/LISA_multi_view/bin/python run_GeoLocLM_all.py \
  --val-dataset-log val_dataset_dupes.log \
  --entity-allowlist val_dataset_dupes_entities.txt \
  --log-dir run_logs_GeoLocLM_dupes \
  --num-top-views 10 --view-nms 0 \
  > run_GeoLocLM_dupes_supervisor.out 2>&1 &
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Seg. VLM checkpoints ----------------------------------------------------
CAD_LISA = os.path.join(ROOT, "/data/1bali/Other_LLM_projects/ECCV_2026/LISA/runs/CAD_LISA_repro20/ckpt_model/global_step5076")
VANILLA_LISA = "vanilla"  # any path WITHOUT 'CAD_LISA' -> base LISA weights

# --- View selectors ----------------------------------------------------------
# The four learned models live in the LISA root (ignore the "copy" ones).
# 'random' and 'GT' are sentinel strings the eval script understands directly.
VIEW_SELECTORS = [
    os.path.join(ROOT, "best_model_view_ranker_cliplora_cross_attention.pt"),
    os.path.join(ROOT, "best_model_view_ranker_cliplora_film.pt"),
    os.path.join(ROOT, "best_model_view_ranker_cliplora_no_fusion.pt"),
    os.path.join(ROOT, "best_model_view_ranker_cliplora_only_clip.pt"),
    "random",
    "GT",
]

# --- Seg. VLMs: (tag, LISA_model_path) ---------------------------------------
LLMS = [
    ("CAD", CAD_LISA),
    ("Vanilla", VANILLA_LISA),
]


def view_selector_tag(vs: str) -> str:
    """Short, filesystem-friendly name for a view-selector spec."""
    if vs in ("random", "GT"):
        return vs
    base = os.path.basename(vs)
    base = base[: -len(".pt")] if base.endswith(".pt") else base
    return base.replace("best_model_view_ranker_cliplora_", "")


def build_jobs(num_top_views: int, view_nms: int, render_target_masks: int = 0,
               val_dataset_log: str = "val_dataset_total.log",
               entity_allowlist: str = None, gt_timeout: int = 20,
               gt_view_offset: int = 0, gt_target_view: int = 0,
               selectors=None, shard_specs=None, llms=None):
    """Return a list of job dicts (Cartesian product of LLMs x view selectors
    x shards).

    `selectors` (optional): iterable of short selector tags (e.g. ['GT']) to keep;
    when given, only those view selectors are launched.
    `llms` (optional): iterable of Seg.VLM tags (e.g. ['CAD']) to keep; when given,
    only those Seg. VLMs are launched.
    `shard_specs` (optional): list of (val_dataset_log_path, shard_tag) tuples.
    When given, each (llm, selector) is fanned out across these shards (each shard
    is an independent process over a disjoint folder subset). When None, a single
    full run over `val_dataset_log` is used."""
    if shard_specs is None:
        shard_specs = [(val_dataset_log, None)]
    jobs = []
    for llm_tag, llm_path in LLMS:
        if llms is not None and llm_tag not in llms:
            continue
        for vs in VIEW_SELECTORS:
            tag = view_selector_tag(vs)
            if selectors is not None and tag not in selectors:
                continue
            for shard_log, shard_tag in shard_specs:
                name = f"LISA-{llm_tag}__{tag}"
                if shard_tag is not None:
                    name = f"{name}__{shard_tag}"
                cmd = [
                    sys.executable,
                    os.path.join(ROOT, "GeoLocLM_exp.py"),
                    "--view_selector_model_path", vs,
                    "--LISA_model_path", llm_path,
                    "--num_top_views", str(num_top_views),
                    "--view_nms", str(view_nms),
                    "--render_target_masks", str(render_target_masks),
                    "--val_dataset_log", shard_log,
                    "--gt_timeout", str(gt_timeout),
                    "--gt_view_offset", str(gt_view_offset),
                    "--gt_target_view", str(gt_target_view),
                ]
                if entity_allowlist:
                    cmd += ["--entity_allowlist", entity_allowlist]
                jobs.append({"name": name, "cmd": cmd})
    return jobs


def make_shard_logs(val_dataset_log, entity_allowlist, num_shards, out_dir):
    """Split the mesh-folder dataset log into `num_shards` balanced shard logs.

    Folders are distributed greedily (largest-first, assign to least-loaded shard)
    so each shard carries roughly equal work. If an entity allowlist is given,
    'work' is the number of allowlisted entities whose cad_name falls in that
    folder; otherwise it is 1 per folder. Returns a list of (shard_log_path,
    shard_tag) tuples. Shard logs are written next to the original log with a
    _shardNN suffix so each run lands in its own _shardNN output dir.
    """
    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    with open(resolve(val_dataset_log)) as fr:
        folders = [ln.strip() for ln in fr if ln.strip()]

    # entity counts per cad_name (basename of folder path)
    counts = {}
    if entity_allowlist:
        with open(resolve(entity_allowlist)) as fr:
            for ln in fr:
                ln = ln.strip()
                if not ln or ln.startswith('#'):
                    continue
                cad = ln.split(',')[0].strip()
                counts[cad] = counts.get(cad, 0) + 1
    weighted = [(os.path.basename(f), f, counts.get(os.path.basename(f), 1))
                for f in folders]
    weighted.sort(key=lambda x: x[2], reverse=True)

    shards = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for _cad, folder, w in weighted:
        si = loads.index(min(loads))
        shards[si].append(folder)
        loads[si] += w

    base = os.path.splitext(os.path.basename(val_dataset_log))[0]
    specs = []
    for si in range(num_shards):
        shard_path = os.path.join(out_dir, f"{base}_shard{si:02d}.log")
        with open(shard_path, 'w') as fw:
            fw.write('\n'.join(shards[si]) + ('\n' if shards[si] else ''))
        specs.append((shard_path, f"shard{si:02d}"))
        print(f"  shard{si:02d}: {len(shards[si]):3d} folders, "
              f"~{loads[si]:4d} entities -> {os.path.relpath(shard_path, ROOT)}")
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-top-views", type=int, default=10)
    parser.add_argument("--view-nms", type=int, default=0)
    parser.add_argument("--val-dataset-log", default="val_dataset_total.log",
                        help="Validation dataset log forwarded to GeoLocLM_exp.py. "
                             "Defaults to the FULL eval set (val_dataset_total.log). "
                             "Pass val_dataset.log for the small 60-CAD subset. The "
                             "log name is appended to each run's output dir so results "
                             "don't collide with other splits.")
    parser.add_argument("--render-target-masks", type=int, default=0,
                        help="Forwarded to GeoLocLM_exp.py. 0 (default) skips the "
                             "slow GT target-mask renders; localization metrics are "
                             "unaffected. Set 1 only when you want the diagnostic 2D "
                             "IoU printouts / inspect_masks overlays.")
    parser.add_argument("--entity-allowlist", default=None,
                        help="Forwarded to GeoLocLM_exp.py --entity_allowlist. Path to a "
                             "file with one 'cad_name,feature,feature_idx' key per line; "
                             "restricts every run to exactly those entities (clean "
                             "single-pass re-run of a specific entity subset).")
    parser.add_argument("--gt-timeout", type=int, default=20,
                        help="Forwarded to GeoLocLM_exp.py --gt_timeout. Hard wall-clock "
                             "timeout (seconds) for the gt_features subprocess. Raise "
                             "(e.g. 120) when recovering heavy meshes that the default "
                             "20s skips.")
    parser.add_argument("--gt-view-offset", type=int, default=0,
                        help="Forwarded to GeoLocLM_exp.py --gt_view_offset. ORACLE (GT) "
                             "rank offset: 0=best view (standard), 1=2nd-best, etc. "
                             "Offset>0 runs land in separate _gtoffset{N} dirs.")
    parser.add_argument("--gt-target-view", type=int, default=0,
                        help="Forwarded to GeoLocLM_exp.py --gt_target_view. If 1, the "
                             "GT oracle is the caption-marked TARGET view (the true view "
                             "oracle the selectors aim to recover) instead of the "
                             "top_views_desc ranking. Top-1 only; lands in separate "
                             "_gttarget dirs.")
    parser.add_argument("--selectors", nargs="+", default=None,
                        help="Optional subset of view-selector tags to launch (e.g. "
                             "'--selectors GT'). Valid tags: cross_attention, film, "
                             "no_fusion, only_clip, random, GT. Default: all six.")
    parser.add_argument("--llms", nargs="+", default=None,
                        help="Optional subset of Seg. VLM tags to launch (e.g. "
                             "'--llms CAD'). Valid tags: CAD, Vanilla. Default: both.")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split the dataset across N parallel shards per (Seg.VLM, "
                             "selector). Each shard is an independent process over a "
                             "disjoint, load-balanced subset of mesh folders, landing "
                             "in its own _shardNN output dir. Total jobs = LLMs x "
                             "selectors x N, round-robined across GPUs. Combine the "
                             "_shardNN dirs afterwards. Default 1 (no sharding).")
    parser.add_argument("--log-dir", default=os.path.join(ROOT, "run_logs_GeoLocLM"))
    parser.add_argument("--stagger-seconds", type=float, default=45.0,
                        help="Delay between consecutive launches so the heavy "
                             "model-load + memory ramp of the runs does not all "
                             "coincide (mitigates host-RAM spikes).")
    parser.add_argument("--poll-seconds", type=float, default=30.0,
                        help="How often the supervisor checks running jobs.")
    parser.add_argument("--max-restarts", type=int, default=1000,
                        help="Per-job auto-restart budget. Each run self-resumes "
                             "from its progress file, so a crash (e.g. OOM) is "
                             "retried from where it left off. 0 disables restarts.")
    parser.add_argument("--cpu-threads", type=int, default=4,
                        help="Cap CPU thread pools (OMP/MKL/OpenBLAS/...) per run "
                             "so 12 concurrent processes don't oversubscribe cores "
                             "and balloon host memory.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the launch plan and exit without running.")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    shard_specs = None
    if args.num_shards > 1:
        print(f"Sharding {args.val_dataset_log} into {args.num_shards} balanced "
              f"shards (by allowlist entity count):")
        shard_dir = os.path.join(ROOT, "dataset_shards")
        os.makedirs(shard_dir, exist_ok=True)
        shard_specs = make_shard_logs(args.val_dataset_log, args.entity_allowlist,
                                      args.num_shards, shard_dir)
        print()

    jobs = build_jobs(args.num_top_views, args.view_nms, args.render_target_masks,
                      val_dataset_log=args.val_dataset_log,
                      entity_allowlist=args.entity_allowlist,
                      gt_timeout=args.gt_timeout,
                      gt_view_offset=args.gt_view_offset,
                      gt_target_view=args.gt_target_view,
                      selectors=args.selectors,
                      shard_specs=shard_specs,
                      llms=args.llms)

    n_sel = len(args.selectors) if args.selectors else len(VIEW_SELECTORS)
    n_llm = len(args.llms) if args.llms else len(LLMS)
    expected = n_llm * n_sel * args.num_shards
    print(f"Total runs: {len(jobs)} (expected {expected})")
    print(f"GPUs: {args.num_gpus}  ->  {len(jobs) / args.num_gpus:.1f} runs/GPU\n")

    # Assign GPUs round-robin so each GPU gets a uniform share of runs.
    for i, job in enumerate(jobs):
        job["gpu"] = i % args.num_gpus
        job["log"] = os.path.join(args.log_dir, f"{job['name']}.gpu{job['gpu']}.log")

    # Show the plan.
    print(f"{'#':>2}  {'GPU':>3}  {'NAME':<28}  LOG")
    for i, job in enumerate(jobs):
        print(f"{i:>2}  {job['gpu']:>3}  {job['name']:<28}  {os.path.relpath(job['log'], ROOT)}")
    print()

    if args.dry_run:
        print("Dry run -- nothing launched.")
        return 0

    def make_env(job):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
        env["PYTHONUNBUFFERED"] = "1"  # keep logs flowing live
        # Cap CPU thread pools so 12 concurrent processes don't oversubscribe the
        # node's cores or balloon resident memory via per-thread arenas.
        t = str(max(1, args.cpu_threads))
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[var] = t
        # glibc malloc: bound per-thread arenas + trim aggressively to return freed
        # memory to the OS (helps keep RSS flat across the long render loop).
        env.setdefault("MALLOC_ARENA_MAX", "2")
        env.setdefault("MALLOC_TRIM_THRESHOLD_", "67108864")
        return env

    def launch(job, append):
        """Start (or restart) a job. `append` keeps prior log history on restart;
        the run self-resumes from its progress file so no work is duplicated."""
        logf = open(job["log"], "a" if append else "w", buffering=1)
        if append:
            logf.write(f"\n===== [supervisor] restart #{job['restarts']} "
                       f"@ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            logf.flush()
        p = subprocess.Popen(
            job["cmd"], cwd=ROOT, env=make_env(job),
            stdout=logf, stderr=subprocess.STDOUT,
        )
        job["proc"] = p
        job["logf"] = logf

    # Launch every run, staggered, each pinned to its GPU.
    start = time.time()
    for i, job in enumerate(jobs):
        job["restarts"] = 0
        job["done"] = False
        job["rc"] = None
        print(f"[GPU {job['gpu']}] launching {job['name']}  ->  {os.path.relpath(job['log'], ROOT)}")
        launch(job, append=False)
        if args.stagger_seconds > 0 and i < len(jobs) - 1:
            time.sleep(args.stagger_seconds)

    print(f"\nLaunched {len(jobs)} runs. Supervising (auto-restart up to "
          f"{args.max_restarts}x/job, self-resuming from progress files). "
          f"Tail any file in {os.path.relpath(args.log_dir, ROOT)} to follow...\n")

    # Supervisor loop: poll, and relaunch any job that died with a non-zero code
    # until it succeeds or exhausts its restart budget.
    while not all(job["done"] for job in jobs):
        time.sleep(args.poll_seconds)
        for job in jobs:
            if job["done"]:
                continue
            rc = job["proc"].poll()
            if rc is None:
                continue  # still running
            job["logf"].close()
            if rc == 0:
                job["rc"] = 0
                job["done"] = True
                print(f"[done] {job['name']:<28} OK")
            elif job["restarts"] < args.max_restarts:
                job["restarts"] += 1
                print(f"[restart] {job['name']:<28} died rc={rc} "
                      f"-> restart {job['restarts']}/{args.max_restarts} (resuming)")
                if args.stagger_seconds > 0:
                    time.sleep(min(args.stagger_seconds, 15.0))
                launch(job, append=True)
            else:
                job["rc"] = rc
                job["done"] = True
                print(f"[give-up] {job['name']:<28} rc={rc} "
                      f"after {job['restarts']} restarts")

    elapsed = time.time() - start

    # Final summary table.
    print("\n==================== SUMMARY ====================")
    failures = 0
    for job in jobs:
        ok = job["rc"] == 0
        failures += 0 if ok else 1
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] GPU{job['gpu']}  {job['name']:<28} "
              f"restarts={job['restarts']}  {os.path.relpath(job['log'], ROOT)}")
    print("================================================")
    print(f"Elapsed: {elapsed/60:.1f} min | "
          f"{len(jobs) - failures}/{len(jobs)} succeeded | logs in {os.path.relpath(args.log_dir, ROOT)}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
