# offline_grad_cos_gpu_minibatch.py
import os, json, math, argparse
from pathlib import Path
import torch

TORCH_DTYPE_MAP = {
    "fp16": torch.float16, "float16": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp32": torch.float32, "float32": torch.float32,
}

def _load_json(p: Path):
    return json.loads(p.read_text())

def _find_rank_dirs(which_dir: Path):
    ranks = sorted([p for p in which_dir.glob("rank_*") if p.is_dir()])
    return ranks if ranks else [which_dir]  # 兼容“没有 rank_* 子目录”的情况

def _load_manifest(rank_dir: Path, which_dir: Path):
    cand_rank = rank_dir / "manifest.json"
    cand_root = which_dir / "manifest.json"
    if cand_rank.exists():
        return _load_json(cand_rank)
    if cand_root.exists():
        return _load_json(cand_root)
    raise FileNotFoundError(f"No manifest.json under {rank_dir} or {which_dir}")

def _safe_to_gpu(t: torch.Tensor, device: torch.device, compute_dtype: torch.dtype):
    if t.device.type != "cpu":
        t = t.cpu()
    if not t.is_pinned():
        try:
            t = t.pin_memory()
        except Exception:
            pass
    return t.to(device=device, dtype=compute_dtype, non_blocking=True)

def _list_minibatch_dirs(step_root: Path):
    # 兼容两种布局：
    # (A) step_xxx/{sft,pg}  —— 老格式（无 mini-batch）
    # (B) step_xxx/batch_idx_*/{sft,pg} —— 新格式
    batch_dirs = sorted([p for p in step_root.glob("batch_idx_*") if p.is_dir()])
    if batch_dirs:
        return batch_dirs
    else:
        # 退化为“单个虚拟 batch”，指向 step 根本身
        return [step_root]

@torch.no_grad()
def _compute_param_cos_between_dirs(
    dir_sft: Path,
    dir_pg: Path,
    dev: torch.device,
    cdtype: torch.dtype,
    do_precond: bool,
    eps: float = 1e-8,
):
    """在给定的两个目录（每个目录下可能有 rank_* 或直接是 tensor 文件）之间，
       逐参数计算 cos/angle/两边范数。返回 (per_param, block_cos_list)。
    """
    rank_dirs_sft = _find_rank_dirs(dir_sft)
    rank_dirs_pg  = _find_rank_dirs(dir_pg)

    mani_sft = _load_manifest(rank_dirs_sft[0], dir_sft)
    mani_pg  = _load_manifest(rank_dirs_pg[0],  dir_pg)
    params_sft = mani_sft["params"]
    params_pg  = mani_pg["params"]

    # 兼容 “flat_param 作为第 0 个”
    flat_param = params_sft[0]; params_sft.append(flat_param); params_sft.pop(0)
    flat_param = params_pg[0];  params_pg.append(flat_param);  params_pg.pop(0)

    assert len(params_sft) == len(params_pg), "SFT/PG params length mismatch"
    for i in range(len(params_sft)):
        assert params_sft[i]["name"] == params_pg[i]["name"], "SFT/PG param order/name mismatch"

    per_param = []
    block_cos = []

    for idx, info in enumerate(params_sft):
        name = info["name"]

        dot_sum  = torch.zeros((), device=dev, dtype=torch.float64)
        n1sq_sum = torch.zeros((), device=dev, dtype=torch.float64)
        n2sq_sum = torch.zeros((), device=dev, dtype=torch.float64)

        for rd_sft, rd_pg in zip(rank_dirs_sft, rank_dirs_pg):
            f_sft = (rd_sft / info["file"]) if (rd_sft / info["file"]).exists() else (dir_sft / info["file"])
            f_pg  = (rd_pg  / params_pg[idx]["file"]) if (rd_pg / params_pg[idx]["file"]).exists() else (dir_pg / params_pg[idx]["file"])

            a_cpu = torch.load(f_sft, map_location="cpu").view(-1)
            b_cpu = torch.load(f_pg , map_location="cpu").view(-1)

            if do_precond:
                vfs = info.get("v_file", None)
                vfp = params_pg[idx].get("v_file", None)
                if vfs and vfp:
                    v_s = torch.load(((rd_sft / vfs) if (rd_sft / vfs).exists() else (dir_sft / vfs)), map_location="cpu").view(-1)
                    v_p = torch.load(((rd_pg  / vfp) if (rd_pg  / vfp).exists() else (dir_pg  / vfp)), map_location="cpu").view(-1)
                    a_cpu = a_cpu / (v_s.sqrt() + eps)
                    b_cpu = b_cpu / (v_p.sqrt() + eps)

            a = _safe_to_gpu(a_cpu, dev, cdtype)
            b = _safe_to_gpu(b_cpu, dev, cdtype)

            dot_sum  += torch.vdot(a, b).to(torch.float64)
            n1sq_sum += torch.vdot(a, a).to(torch.float64)
            n2sq_sum += torch.vdot(b, b).to(torch.float64)

            del a_cpu, b_cpu, a, b
            torch.cuda.synchronize(dev)

        n1 = torch.sqrt(torch.clamp(n1sq_sum, min=1e-30))
        n2 = torch.sqrt(torch.clamp(n2sq_sum, min=1e-30))
        cos = torch.clamp(dot_sum / (n1 * n2 + 1e-12), min=-1+1e-6, max=1-1e-6)
        ang = torch.rad2deg(torch.acos(cos))

        per_param.append({
            "name": name,
            "cos": float(cos.item()),
            "angle_deg": float(ang.item()),
            "norm_sft": float(n1.item()),
            "norm_pg":  float(n2.item()),
        })
        block_cos.append(float(cos.item()))

    return per_param, block_cos, params_sft, params_pg

@torch.no_grad()
def _accumulate_avg_tensors_over_minibatches(
    step_root: Path,
    params_sft_meta,
    params_pg_meta,
    do_precond: bool,
    eps: float,
):
    """逐 mini-batch、逐 rank 把 tensor 读到 CPU 上做求和，
       返回：avg_sft_files[name] 与 avg_pg_files[name] 两个“CPU Tensor”的字典（已经是均值）。
       说明：为避免巨量显存占用，这里全程在 CPU 上累计。
    """
    batch_dirs = _list_minibatch_dirs(step_root)

    # 初始化累加器
    avg_sft = {}
    avg_pg  = {}
    count   = 0  # 统计 batch*rank 的份数

    for bdir in batch_dirs:
        dir_sft = bdir / "sft" if (bdir / "sft").exists() else step_root / "sft"
        dir_pg  = bdir / "pg"  if (bdir / "pg").exists()  else step_root / "pg"

        rank_dirs_sft = _find_rank_dirs(dir_sft)
        rank_dirs_pg  = _find_rank_dirs(dir_pg)

        for rd_sft, rd_pg in zip(rank_dirs_sft, rank_dirs_pg):
            count += 1
            for idx, info in enumerate(params_sft_meta):
                name = info["name"]
                file_sft = info["file"]
                file_pg  = params_pg_meta[idx]["file"]

                f_sft = (rd_sft / file_sft) if (rd_sft / file_sft).exists() else (dir_sft / file_sft)
                f_pg  = (rd_pg  / file_pg ) if (rd_pg  / file_pg ).exists() else (dir_pg  / file_pg )

                a = torch.load(f_sft, map_location="cpu").view(-1).to(torch.float64)
                b = torch.load(f_pg , map_location="cpu").view(-1).to(torch.float64)

                if do_precond:
                    vfs = info.get("v_file", None)
                    vfp = params_pg_meta[idx].get("v_file", None)
                    if vfs and vfp:
                        vf_s = ((rd_sft / vfs) if (rd_sft / vfs).exists() else (dir_sft / vfs))
                        vf_p = ((rd_pg  / vfp) if (rd_pg  / vfp).exists() else (dir_pg  / vfp))
                        vs = torch.load(vf_s, map_location="cpu").view(-1).to(torch.float64)
                        vp = torch.load(vf_p, map_location="cpu").view(-1).to(torch.float64)
                        a = a / (vs.sqrt() + eps)
                        b = b / (vp.sqrt() + eps)

                if name not in avg_sft:
                    avg_sft[name] = torch.zeros_like(a, dtype=torch.float64, device="cpu")
                    avg_pg[name]  = torch.zeros_like(b, dtype=torch.float64, device="cpu")
                avg_sft[name].add_(a)
                avg_pg[name].add_(b)

                del a, b

    if count == 0:
        raise RuntimeError("No gradients found to average.")

    # 取平均
    for k in avg_sft.keys():
        avg_sft[k].div_(count)
        avg_pg[k].div_(count)

    return avg_sft, avg_pg

@torch.no_grad()
def compute_step_stats_gpu_with_minibatch(
    dump_root: str,
    step: int,
    device: str = "cuda:0",
    compute_dtype: str = "fp32",
    do_precond: bool = False,
    eps: float = 1e-8,
    topk: int = 10,
):
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(device)
    cdtype = TORCH_DTYPE_MAP.get(compute_dtype, torch.float32)

    step_root = Path(dump_root) / f"step_{step:03d}"
    # 新格式下的最小单位：batch 目录
    batch_dirs = _list_minibatch_dirs(step_root)

    # ========= 1) 逐 mini-batch 的分析 =========
    print(f"[step {step}] Found {len(batch_dirs)} mini-batch dir(s).")
    ref_params_sft = None
    ref_params_pg  = None

    for bdir in batch_dirs:
        dir_sft = bdir / "sft" if (bdir / "sft").exists() else step_root / "sft"
        dir_pg  = bdir / "pg"  if (bdir / "pg").exists()  else step_root / "pg"
        assert dir_sft.exists() and dir_pg.exists(), f"Missing {dir_sft} or {dir_pg}"

        per_param, block_cos, params_sft_meta, params_pg_meta = _compute_param_cos_between_dirs(
            dir_sft, dir_pg, dev, cdtype, do_precond, eps
        )

        if ref_params_sft is None:
            ref_params_sft = params_sft_meta
        if ref_params_pg is None:
            ref_params_pg = params_pg_meta

        if block_cos:
            gb = sum(block_cos) / len(block_cos)
            t = torch.tensor(gb, device=dev).clamp(-1+1e-6, 1-1e-6)
            gb_ang = float(torch.rad2deg(torch.acos(t)).item())
        else:
            gb, gb_ang = None, None

        tag = bdir.name if bdir != step_root else "batch_idx_000"  # 老格式时给个占位名
        print(f"[step {step} | {tag}] global_block_cos={gb:.4f}  global_block_ang_deg={gb_ang:.2f}")
        for r in per_param:
            print(f"  {r['name']:<70} cos={r['cos']:+.4f}  angle={r['angle_deg']:6.2f}°  "
                  f"||sft||={r['norm_sft']:.3e}  ||pg||={r['norm_pg']:.3e}      sft/pg norm={r['norm_sft']/r['norm_pg']:.3f}")
        if topk and per_param:
            top = sorted(per_param, key=lambda x: x["angle_deg"], reverse=True)[:topk]
            print("  Top-|cos| layers:")
            for r in top:
                print(f"    {r['name']:<68} cos={r['cos']:+.4f}  angle={r['angle_deg']:6.2f}°")

    # ========= 2) mini-batch 平均梯度后的“总分析” =========
    # 在 CPU 上把 (batch, rank) 维度全部平均成一份 SFT 与一份 PG，再做一次 cos/angle。
    avg_sft, avg_pg = _accumulate_avg_tensors_over_minibatches(
        step_root, ref_params_sft, ref_params_pg, do_precond, eps
    )

    per_param_total = []
    block_cos_total = []

    for info_sft, info_pg in zip(ref_params_sft, ref_params_pg):
        name = info_sft["name"]

        a_cpu = avg_sft[name]  # float64 on CPU
        b_cpu = avg_pg[name]

        # 放到 GPU 做乘加
        a = _safe_to_gpu(a_cpu, dev, cdtype)
        b = _safe_to_gpu(b_cpu, dev, cdtype)

        dot  = torch.vdot(a, b).to(torch.float64)
        n1sq = torch.vdot(a, a).to(torch.float64)
        n2sq = torch.vdot(b, b).to(torch.float64)

        n1 = torch.sqrt(torch.clamp(n1sq, min=1e-30))
        n2 = torch.sqrt(torch.clamp(n2sq, min=1e-30))
        cos = torch.clamp(dot / (n1 * n2 + 1e-12), min=-1+1e-6, max=1-1e-6)
        ang = torch.rad2deg(torch.acos(cos))

        per_param_total.append({
            "name": name,
            "cos": float(cos.item()),
            "angle_deg": float(ang.item()),
            "norm_sft": float(n1.item()),
            "norm_pg":  float(n2.item()),
        })
        block_cos_total.append(float(cos.item()))

        del a, b

    if block_cos_total:
        gb = sum(block_cos_total) / len(block_cos_total)
        t = torch.tensor(gb, device=dev).clamp(-1+1e-6, 1-1e-6)
        gb_ang = float(torch.rad2deg(torch.acos(t)).item())
    else:
        gb, gb_ang = None, None

    print(f"[step {step} | mean_of_minibatches] global_block_cos={gb:.4f}  global_block_ang_deg={gb_ang:.2f}")
    for r in per_param_total:
        print(f"  {r['name']:<70} cos={r['cos']:+.4f}  angle={r['angle_deg']:6.2f}°  "
              f"||sft||={r['norm_sft']:.3e}  ||pg||={r['norm_pg']:.3e}      sft/pg norm={r['norm_sft']/r['norm_pg']:.3f}")
    if topk and per_param_total:
        top = sorted(per_param_total, key=lambda x: x["angle_deg"], reverse=True)[:topk]
        print("  Top-|cos| layers (mean-of-minibatches):")
        for r in top:
            print(f"    {r['name']:<68} cos={r['cos']:+.4f}  angle={r['angle_deg']:6.2f}°")

    return {
        "per_minibatch_done": True,
        "mean_of_minibatches": {"global_block_cos": gb, "global_block_ang_deg": gb_ang}
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_root", type=str, default="./grad_dumps/on_pg_sft")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--compute_dtype", type=str, default="fp32", choices=list(TORCH_DTYPE_MAP.keys()))
    ap.add_argument("--precond", action="store_true", help="use preconditioned cosine if v_file is present")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    compute_step_stats_gpu_with_minibatch(
        dump_root=args.dump_root,
        step=args.step,
        device=args.device,
        compute_dtype=args.compute_dtype,
        do_precond=args.precond,
        topk=args.topk,
    )

if __name__ == "__main__":
    main()
