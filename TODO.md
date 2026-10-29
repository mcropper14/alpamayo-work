# Real-Time E2E Driving in AlpaSim — TODO

Proposal: [E2E_Proposal-updated.pdf](E2E_Proposal-updated.pdf)

Goal: reproducible closed-loop baselines for SIMFORGE-D1.5, Alpamayo 1.5, FlashDrive
and VaVAM, then a visual feature cache (cv-cache) for Alpamayo evaluated against them.

---

## Phase 1: Robust Baseline Benchmarking

Establish a reproducible benchmarking framework within AlpaSim before any architectural
modifications.

### 1.0 What we already know

- [x] VaVAM single-scene rollout works end-to-end on Bridges-2 (Singularity + host
      runtime, `run_vavam.sbatch`, scene `clipgt-02eadd92`, NRE 26.04).
- **SIMFORGE-D1.5** (`SIMFORGE-AI/SIMFORGE-D1.5`, gated on HF) is a 7.21B distillation of
  Alpamayo 1.5 with `model_type: alpamayo1_5` → should run on the existing
  `driver=alpamayo1_5` with a different `checkpoint_path`.
- **FlashDrive** (github.com/z-lab/flashdrive) is *not* a separate model: it is a compiled,
  asynchronous serving of Alpamayo 1.5 weights. (The HF "FlashVLA" checkpoints are
  LIBERO/RoboTwin robotics models, not driving.)
- SIMFORGE's model card has a reference benchmark ("NuRec96", A100 40GB,
  `e2e_challenge=dev`, 1 rollout/scene, 200 steps, 4×1080p cameras) — use it as a sanity
  check, not as ground truth:
  - Synchronous model latency: A1.5 ≈ 1.69 s, SIMFORGE ≈ 1.56 s, FlashDrive ≈ 1.25 s
  - Mean `scene_score`: A1.5 0.650, SIMFORGE 0.623, FlashDrive 0.722
- **Async drivers make RPC latency meaningless as a model-speed metric.** SIMFORGE and
  FlashDrive return the latest cached plan in ~3 ms while the model runs in the background.
  We must record *model* latency (inside the driver) separately from *RPC* latency.
- All baselines are >10× over the 100 ms budget. The cv-cache alone will not close that
  gap; it is one component of the latency breakdown to measure.

### 1.1 Infrastructure

Harness lives in [benchmark/](benchmark/). Workflow (repo root):

```bash
export HF_TOKEN=...                          # login node
bash benchmark/download_weights.sh           # once: A1.5, SIMFORGE, Qwen configs → data/hf_cache
bash benchmark/prepare.sh vavam              # per arm: downloads scenes, writes benchmark/runs/<arm>/
sbatch benchmark/run_benchmark.sbatch vavam  # per arm: one L40S job
uv run python benchmark/summarize.py         # → table + benchmark/results.csv
```

- [x] Generalize `run_vavam.sbatch` → `benchmark/run_benchmark.sbatch ARM`; arms defined in
      `benchmark/env.sh` (`vavam`, `alpamayo1_5`, `simforge_d1_5`).
- [x] GPU: L40S-48 for all latency numbers.
- [x] Record per run (`benchmark-env.yaml`): git commit, dirty flag, GPU, driver, torch/CUDA.
- [ ] Run `download_weights.sh` (needs `HF_TOKEN`; HF cache moved to `/ocean`, the 25 GiB
      home quota cannot hold the weights).
- [ ] Check that the renderer and Alpamayo 1.5 (~22 GiB) both fit on one 48 GiB L40S.

### 1.2 Scene set

- [x] Dev set: 8 scenes from `public_2601`, seed-0 sample → `benchmark/scenes_dev.txt`
      (each USDZ ≈ 1.8 GB; downloaded by the first `prepare.sh`).
- [ ] Final set: larger fixed subset; freeze the scene-ID list in a CSV checked into the repo.
- [ ] Pre-download all scenes on a login node; stage them in one directory used by
      renderer, physics and runtime (no duplicate UUIDs).
- [x] 3 rollouts/scene (`N_ROLLOUTS` in `benchmark/env.sh`).
- [ ] Sim settings differ per arm by design: VaVAM is a 1-camera 2 Hz model (500 ms
      control step); the Alpamayo family runs 4 cameras at 10 Hz. Report this alongside the
      closed-loop scores; the 100 ms check uses measured model latency, not the control rate.

### 1.3 Model integration

- [x] VaVAM (`driver=vavam`) — real-time reference policy.
- [ ] Alpamayo 1.5 (`driver=alpamayo1_5` + `exp/presets/alpamayo1_5_comparison_4cam`).
      **Open issue:** two runs produced NaN inside the model (finite inputs) on scene
      `060131e7` rollouts 2–3 after ~730 inferences; the NaN reached `torch.multinomial`
      and its device assert killed the driver. Now caught per stage
      (`NonFiniteModelOutputError`) with inputs dumped to `driver/nonfinite_inputs/` —
      replay a dump to tell input-specific numerics from GPU state corruption.
      Failed runs kept in `benchmark/failed_runs/`.
- [ ] SIMFORGE-D1.5 (`driver=alpamayo1_5`, `model.checkpoint_path=<SIMFORGE path>`);
      verify the reduced architecture loads with `Alpamayo15Model`.
- [ ] FlashDrive (github.com/z-lab/flashdrive, `z-lab/Alpamayo-1.5-10B` + `-PARO` + `-DFlash`):
      no AlpaSim integration upstream. Plan: new driver model calling
      `flashdrive.from_pretrained(...).sample_trajectories_streaming(data)`, in its own uv env
      (vLLM Marlin kernels, CUDA 12.8). Stage hooks won't see inside its CUDA graphs → total
      latency only. Its README reports A1.5 717 → 151 ms on an RTX PRO 6000.
      **Overlap with Phase 2:** FlashDrive's "streaming inference" already reuses each
      frame's KV cache across windows — the cv-cache must be positioned against it.
- [ ] (Optional) Alpamayo R1 (`driver=alpamayo1`) for the reasoning-trace overhead comparison.

### 1.4 Metrics

To measure whether models satisfy the 100 ms inference budget (10 Hz control loop):

All written by the harness and summarized by `benchmark/summarize.py`:

| Metric | Source |
|---|---|
| Model latency per inference (ms), median/mean/P95 | `driver/latency.jsonl` `inference_ms` (CUDA-synced, inside driver) |
| Component latency: preprocess, vision encoder, LLM prefill, CoT decode (+ token count), trajectory (diffusion) | `latency.jsonl`, forward hooks on `vlm` / `vlm.model.visual` (Alpamayo family) |
| % inference cycles ≤ 100 ms | `summarize.py` from `inference_ms` |
| RPC latency per `Drive` call (server side) and queue wait | `latency.jsonl` `rpc_ms`, `queue_wait_ms` |
| Simulator throughput | `rollouts/*/*/timing.json` `sim_steps_per_s`, `realtime_ratio` |
| Peak GPU memory | `latency.jsonl` `peak_reserved_gib` |
| Route completion | eval `progress_rel` |
| Collision rate | eval `collision_at_fault`, `collision_any` |
| Overall driving score | eval `scene_score` mean + pass rate |

- [x] Exclude warmup inferences (`--warmup`, default 3 per driver process); report
      median, mean, P95.
- [x] One aggregation script → `benchmark/summarize.py`.
- [ ] VaVAM batches up to 32 requests across concurrent rollouts: its `inference_ms` is per
      batch (`mean_batch_size` in the table). Decide whether to compare at batch size 1.
- [ ] Sanity-check Alpamayo 1.5 synchronous latency against SIMFORGE's ~1.69 s (A100).

---

## Phase 2: Visual Feature Caching Implementation

Consecutive driving frames contain substantial temporal redundancy; repeatedly processing
similar visual information through deep transformer architectures introduces unnecessary
latency. This phase engineers the caching mechanism for Alpamayo.

Implemented as a separate module: [visual_cache/](src/driver/src/alpasim_driver/visual_cache/)
(config: `driver.model.visual_cache`, off by default).

Redundancy found in the stock pipeline: each inference feeds 4 cameras × 4 frames, and
`generate` repeats them once per trajectory sample (6) → the encoder runs on **96 images
per step, of which 4 are new**. Qwen3-VL encodes each image independently (attention is
restricted to the image's own patches), so per-image reuse is exact.

- [x] **Encoder output storage:** `VisualFeatureCache` — LRU map from a 64-bit pixel
      fingerprint to the image's merged visual tokens + deepstack features.
- [x] **Redundancy bypassing:** `CachedImageEncoder` replaces `Qwen3VLModel.get_image_features`
      on the loaded model; only unique, uncached images reach the encoder, in one batch.
- [x] **State updating:** strategy `exact` (bit-identical frames only, lossless) or
      `near_duplicate` (reuse the most similar *encoded* frame if mean |Δpixel| ≤ threshold;
      each encoded frame serves ≤ `max_near_duplicate_uses` others → periodic refresh).
- [x] Per-inference counters in `latency.jsonl` (`vision_images`, `vision_encoded`,
      `vision_exact_hits`, `vision_near_duplicate_hits`) and in `summarize.py`.
- [ ] Run `sbatch benchmark/visual_cache_microbench.sbatch`: isolated encoder latency,
      images encoded per step, and max deviation vs. uncached features (bf16 kernel noise).
- [ ] Closed loop: `prepare.sh` + `run_benchmark.sbatch` for `alpamayo1_5_vcache_exact`
      (prepared) and `alpamayo1_5_vcache_neardup`; compare against `alpamayo1_5`.
- [ ] Expectation check: the cache only removes vision-encoder work (FlashDrive measures
      encode ≈ 87 of 717 ms on A1.5); LLM prefill still processes all visual tokens. The
      next lever is reusing prefill KV for unchanged frames — overlaps FlashDrive streaming.

## Phase 3: Ablation Studies and Optimization

Optimize the memory–latency tradeoff without degrading spatial awareness or trajectory
safety.

Ablation variables (all are `driver.model.visual_cache.*` overrides; add arms to
`benchmark/env.sh`):
- [ ] Cache sizes — `max_entries` (needs ≥ 16 per interleaved rollout for full reuse;
      runtime runs 4 rollouts concurrently → default 64).
- [ ] Update strategies — `strategy` (`exact` vs `near_duplicate`),
      `near_duplicate_threshold`, `max_near_duplicate_uses` (refresh period).

Targeted measurements:
- [ ] Reduction in isolated vision-encoder latency from caching.
- [ ] GPU memory utilization stays within hardware limits.
- [ ] Closed-loop performance vs. Phase 1 baselines (safety and route completion preserved).
