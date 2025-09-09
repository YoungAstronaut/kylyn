# offline_grad_cos_gpu.py
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
    # 兼容“没有 rank 子目录”的情况：直接把 which_dir 当 rank_000
    return ranks if ranks else [which_dir]

def _load_manifest(rank_dir: Path, which_dir: Path):
    # 优先 rank 目录里的 manifest.json；否则退回 which 根下的 manifest.json
    cand_rank = rank_dir / "manifest.json"
    cand_root = which_dir / "manifest.json"
    if cand_rank.exists():
        return _load_json(cand_rank)
    if cand_root.exists():
        return _load_json(cand_root)
    raise FileNotFoundError(f"No manifest.json under {rank_dir} or {which_dir}")

def _safe_to_gpu(t: torch.Tensor, device: torch.device, compute_dtype: torch.dtype):
    # pin + non_blocking 推荐；从 torch.load 读出的 tensor 不是 pinned，手动 pin 一下
    if t.device.type != "cpu":
        t = t.cpu()
    if not t.is_pinned():
        try:
            t = t.pin_memory()
        except Exception:
            pass
    return t.to(device=device, dtype=compute_dtype, non_blocking=True)

@torch.no_grad()
def compute_step_stats_gpu(
    dump_root: str,
    step: int,
    device: str = "cuda:0",
    compute_dtype: str = "fp32",       # 计算精度（建议 fp32）
    do_precond: bool = False,          # 是否做预条件余弦（需要 v_file）
    eps: float = 1e-8,
    topk: int = 10,                    # 输出|cos|最大的若干层
):
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(device)
    cdtype = TORCH_DTYPE_MAP.get(compute_dtype, torch.float32)

    base = Path(dump_root) / f"step_{step:07d}"
    dir_sft = base / "sft"
    dir_pg  = base / "pg"
    assert dir_sft.exists() and dir_pg.exists(), f"Missing {dir_sft} or {dir_pg}"

    # 发现 rank 目录
    rank_dirs_sft = _find_rank_dirs(dir_sft)
    rank_dirs_pg  = _find_rank_dirs(dir_pg)

    # 用第一个 rank 的 manifest 作为基准（只取 params 列表）
    mani_sft = _load_manifest(rank_dirs_sft[0], dir_sft)
    mani_pg  = _load_manifest(rank_dirs_pg[0],  dir_pg)
    params_sft = mani_sft["params"]
    params_pg  = mani_pg["params"]
    flat_param = params_sft[0]
    params_sft.append(flat_param)
    params_sft.pop(0)
    flat_param = params_pg[0]
    params_pg.append(flat_param)
    params_pg.pop(0)
    assert len(params_sft) == len(params_pg), "SFT/PG params length mismatch"
    for i in range(len(params_sft)):
        assert params_sft[i]["name"] == params_pg[i]["name"], "SFT/PG param order/name mismatch"

    per_param = []
    block_cos = []

    for idx, info in enumerate(params_sft):
        name = info["name"]

        # 全局累加（放在 GPU 上，避免 CPU 反复转型）
        dot_sum  = torch.zeros((), device=dev, dtype=torch.float64)
        n1sq_sum = torch.zeros((), device=dev, dtype=torch.float64)
        n2sq_sum = torch.zeros((), device=dev, dtype=torch.float64)

        for rd_sft, rd_pg in zip(rank_dirs_sft, rank_dirs_pg):
            # 文件路径（兼容 manifest 在 root 的情况）
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
                else:
                    # 没有 v_file 就退化为普通余弦
                    pass

            # 搬到 GPU 做乘加
            a = _safe_to_gpu(a_cpu, dev, cdtype)
            b = _safe_to_gpu(b_cpu, dev, cdtype)

            # 向量内积/范数平方（fp64 累加更稳）
            dot_sum  += torch.vdot(a, b).to(torch.float64)
            n1sq_sum += torch.vdot(a, a).to(torch.float64)
            n2sq_sum += torch.vdot(b, b).to(torch.float64)

            # 释放显存
            del a_cpu, b_cpu, a, b
            torch.cuda.synchronize(dev)  # 可选：更稳的峰值控制；追求吞吐可注释

        # 计算本层 cos/angle
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

    # 全局 block-normalized（逐层 cos 的平均）
    if block_cos:
        gb = sum(block_cos) / len(block_cos)
        t = torch.tensor(gb, device=dev).clamp(-1+1e-6, 1-1e-6)
        gb_ang = float(torch.rad2deg(torch.acos(t)).item())
    else:
        gb, gb_ang = None, None

    # 打印摘要 & TopK
    print(f"[step {step}] global_block_cos={gb:.4f}  global_block_ang_deg={gb_ang:.2f}")
    for r in per_param:
        print(f"  {r['name']:<70} cos={r['cos']:+.4f}  angle={r['angle_deg']:6.2f}°  "
              f"||sft||={r['norm_sft']:.3e}  ||pg||={r['norm_pg']:.3e}      sft/pg norm={r['norm_sft']/r['norm_pg']}")
    if topk and per_param:
        top = sorted(per_param, key=lambda x: x["angle_deg"], reverse=True)[:topk]
        print("Top-|cos| layers:")
        for r in top:
            print(f"  {r['name']:<70} cos={r['cos']:+.4f}  angle={r['angle_deg']:6.2f}°  "
                  f"||sft||={r['norm_sft']:.3e}  ||pg||={r['norm_pg']:.3e}")

    return per_param, {"global_block_cos": gb, "global_block_ang_deg": gb_ang}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_root", type=str, default="./grad_dumps")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--compute_dtype", type=str, default="fp32", choices=list(TORCH_DTYPE_MAP.keys()))
    ap.add_argument("--precond", action="store_true", help="use preconditioned cosine if v_file is present")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    compute_step_stats_gpu(
        dump_root=args.dump_root,
        step=args.step,
        device=args.device,
        compute_dtype=args.compute_dtype,
        do_precond=args.precond,
        topk=args.topk,
    )

if __name__ == "__main__":
    main()
