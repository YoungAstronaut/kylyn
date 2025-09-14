from typing import List, Callable, Optional, Dict, Any
import math
import statistics

def _cos(a, b):
    if a is None or b is None: return 0.0
    da = math.sqrt(sum(x*x for x in a)); db = math.sqrt(sum(x*x for x in b))
    if da == 0 or db == 0: return 0.0
    return sum(x*y for x, y in zip(a, b)) / (da*db)

def _zscore(x, xs):
    if not xs: return 0.0
    mu = statistics.fmean(xs); sd = statistics.pstdev(xs) or 1e-6
    return (x - mu) / sd

def detect_irrelevant(
    block_text: str,
    *,
    query_text: Optional[str],
    reference_texts: List[str],
    embed_fn: Callable[[str], List[float]], # 嵌入函数，返回同维度向量
    cross_encoder_fn: Optional[Callable[[str, str], float]] = None, # 交叉编码器打分函数（输入 q, ref, 或 block, ref）
    nli_fn: Optional[Callable[[str, str], Dict[str, float]]] = None, # NLI 函数，返回 {"entail":p1,"neutral":p2,"contradict":p3}
    ppl_fn: Optional[Callable[[str, Optional[str]], float]] = None, # LM困惑度函数。返回给定文本的 per-token ppl 或 loss
    bm25_fn: Optional[Callable[[str, str], float]] = None, # BM25/词法覆盖打分函数
    batch_neg_refs: Optional[List[str]] = None, # 批内负采样：用于自适应阈值（传入同batch其他参考作为neg）
    # 阈值与融合
    th_cos_z: float = -0.2,      # 低于此 z 视为低相似
    th_ce_z: float = -0.2,
    th_bm25_z: float = -0.2,
    th_ppl_drop_z: float = -0.2, # ΔPPL 的 z 分数
    topk: int = 5
) -> Dict[str, Any]:
    """
    返回：
      {
        "is_irrelevant": bool,
        "signals": {...},   # 各分数/阈值/尖锐度
        "label": "irrelevant/related/conflict",
        "why": str          # 决策说明
      }
    """
    refs = reference_texts or []
    if not refs:
        return {"is_irrelevant": True,
                "signals": {}, "label": "irrelevant",
                "why": "no reference provided"}

    # 1) 轻量信号：cos / bm25
    v_blk = embed_fn(block_text)
    cos_scores, bm25_scores = [], []
    for r in refs:
        vr = embed_fn(r)
        cos_scores.append(_cos(v_blk, vr))
        if bm25_fn: bm25_scores.append(bm25_fn(block_text, r))

    # 2) 重模型（可选）：交叉编码器 + NLI
    ce_scores = []
    nli_conflict_prob = 0.0
    if cross_encoder_fn:
        for r in refs:
            ce_scores.append(cross_encoder_fn(block_text, r))  # 或 (query_text, r)
    if nli_fn and query_text:
        # 用“(block_text -> query_text)”与“(query_text -> block_text)”两向评估取max矛盾
        p1 = nli_fn(block_text, query_text).get("contradict", 0.0)
        p2 = nli_fn(query_text, block_text).get("contradict", 0.0)
        nli_conflict_prob = max(p1, p2)

    # 3) PPL 下降（上下文有用性）
    # ΔPPL = ppl(no_ctx) - ppl(with_ctx_top1)；越大越相关
    ppl_drop = None
    if ppl_fn:
        try:
            noctx = ppl_fn(block_text, None)
            # 取 top-1 相似参考拼接为上下文
            top1_idx = max(range(len(refs)), key=lambda i: cos_scores[i])
            withctx = ppl_fn(block_text, refs[top1_idx])
            ppl_drop = (noctx - withctx)
        except Exception:
            ppl_drop = None

    # 4) 多参考聚合与尖锐度
    def take_topk(xs):
        xs_sorted = sorted(xs, reverse=True)
        return xs_sorted[:min(len(xs_sorted), topk)]

    topk_cos = take_topk(cos_scores)
    agg_cos = max(cos_scores)
    sharp_cos = (agg_cos / (sum(topk_cos)/len(topk_cos))) if topk_cos else 0.0

    agg_bm25 = (max(bm25_scores) if bm25_scores else None)
    if ce_scores:
        topk_ce = take_topk(ce_scores)
        agg_ce = max(ce_scores); sharp_ce = agg_ce / (sum(topk_ce)/len(topk_ce))
    else:
        agg_ce, sharp_ce = None, None

    # 5) 批内负对比 -> z 分数
    neg_cos = []
    if batch_neg_refs:
        for r in batch_neg_refs:
            neg_cos.append(_cos(v_blk, embed_fn(r)))
    z_cos = _zscore(agg_cos, neg_cos) if neg_cos else 0.0

    neg_bm25 = []
    if bm25_fn and batch_neg_refs:
        for r in batch_neg_refs:
            neg_bm25.append(bm25_fn(block_text, r))
    z_bm25 = _zscore(agg_bm25, neg_bm25) if (bm25_fn and agg_bm25 is not None and neg_bm25) else 0.0

    z_ce = 0.0
    if cross_encoder_fn and ce_scores and batch_neg_refs:
        neg_ce = [cross_encoder_fn(block_text, r) for r in batch_neg_refs]
        z_ce = _zscore(agg_ce, neg_ce) if neg_ce else 0.0

    z_ppl = 0.0
    if ppl_drop is not None and batch_neg_refs:
        # 近似负例 ppl_drop，随便取若干 neg 的参考当上下文看是否“帮助”
        neg_ppl_drops = []
        for r in batch_neg_refs[:min(8, len(batch_neg_refs))]:
            try:
                noctx = ppl_fn(block_text, None)
                withctx = ppl_fn(block_text, r)
                neg_ppl_drops.append(noctx - withctx)
            except Exception:
                continue
        z_ppl = _zscore(ppl_drop, neg_ppl_drops) if neg_ppl_drops else 0.0

    # 6) 规则决策（简单稳健）
    reasons = []
    conflict = (nli_conflict_prob >= 0.6)
    if conflict:
        reasons.append(f"NLI_conflict={nli_conflict_prob:.2f}")

    low_sim = (z_cos < th_cos_z) and ((not ce_scores) or (z_ce < th_ce_z))
    weak_lex = (bm25_fn is None) or (z_bm25 < th_bm25_z)
    weak_help = (ppl_drop is None) or (z_ppl < th_ppl_drop_z)
    low_sharp = (sharp_cos < 1.05) and ((sharp_ce is None) or (sharp_ce < 1.05))

    is_irrel = (conflict) or (low_sim and weak_lex) or (low_sim and weak_help) or (low_sharp and low_sim)

    label = "irrelevant" if is_irrel else "related"
    if conflict and not is_irrel: label = "conflict"

    reasons.append(
        f"z_cos={z_cos:.2f}, z_ce={z_ce:.2f}, z_bm25={z_bm25:.2f}, z_ppl={z_ppl:.2f}, "
        f"sharp_cos={sharp_cos:.2f}" + (f", sharp_ce={sharp_ce:.2f}" if sharp_ce is not None else "")
    )
    return {
        "is_irrelevant": bool(is_irrel),
        "signals": {
            "agg_cos": agg_cos, "z_cos": z_cos, "sharp_cos": sharp_cos,
            "agg_ce": agg_ce, "z_ce": z_ce, "sharp_ce": sharp_ce,
            "agg_bm25": agg_bm25, "z_bm25": z_bm25,
            "ppl_drop": ppl_drop, "z_ppl": z_ppl,
            "nli_conflict": nli_conflict_prob
        },
        "label": "conflict" if conflict else label,
        "why": "; ".join(reasons)
    }