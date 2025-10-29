import colorsys
import re
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any, Tuple
import math
import statistics
import torch
import torch.nn.functional as F

from recipe.mixed_train.calculate_similarity import detect_irrelevant

PUNCTS = r'.!?;:,，。！？；：'  # 含中英文常见标点
CLS = re.escape(PUNCTS)
PAT = re.compile(rf'(?:[^{CLS}\r\n]*?(?:[{CLS}]|\r?\n)+[ \t]*)|(?:[^{CLS}\r\n]+$)')

@dataclass
class Block:
    start: int
    end: int            # exclusive
    peak_idx: int
    mean_entropy: float
    max_entropy: float
    sim_to_query: Optional[float] = None
    sim_to_refs: Optional[float] = None
    trigger_sft: bool = False
    reason: str = ""    # why trigger / not trigger
    meta: Dict[str, Any] = None
    text: Optional[str] = None

def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = statistics.fmean(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, sd


def _mean_std_tensor(xs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """张量版本的均值标准差计算"""
    # 使用 mask 过滤有效值
    valid_xs = xs[mask]
    if valid_xs.numel() == 0:
        return torch.tensor(0.0, device=xs.device), torch.tensor(0.0, device=xs.device)

    m = torch.mean(valid_xs)
    # 计算总体标准差（与 statistics.pstdev 对应）
    if valid_xs.numel() > 1:
        sd = torch.std(valid_xs, unbiased=False)
    else:
        sd = torch.tensor(0.0, device=xs.device)
    return m, sd

def _gaussian_like_smooth_tensor(xs: torch.Tensor, w: int = 3) -> torch.Tensor:
    """张量版本的平滑操作"""
    if w <= 1 or xs.size(-1) < 3:
        return xs.clone()

    # 确保窗口大小为奇数
    w = max(3, w | 1)
    r = w // 2

    # 使用一维平均池化进行平滑
    # 需要添加批次和通道维度以适应卷积操作
    xs_4d = xs.unsqueeze(1)  # [bsz, 1, length]

    # 创建平均池化核
    padding = r
    smoothed = F.avg_pool1d(xs_4d, kernel_size=w, stride=1, padding=padding)

    # 移除添加的维度
    return smoothed.squeeze(1)

def _gaussian_like_smooth(xs: List[float], w: int = 3) -> List[float]:
    """Simple symmetric box smoothing (odd window)."""
    if w <= 1 or len(xs) < 3:
        return xs[:]
    w = max(3, w | 1)  # make odd
    r = w // 2
    out = []
    for i in range(len(xs)):
        lo, hi = max(0, i - r), min(len(xs), i + r + 1)
        out.append(statistics.fmean(xs[lo:hi]))
    return out

def _is_sentence_boundary(tok: str) -> bool:
    # 轻量句读边界；你可按语种扩充
    return tok in {".", "!", "?", ",", ";", ":", "。", "！", "？", "，", "；", "：", "\n"}

def _is_connective(tok: str) -> bool:
    # 语义连接词：扩展这个表对中文/英文都有效
    return tok.lower() in {
        "however","but","therefore","thus","so","because","then","hence","while","although","though",
        "因此","但是","然而","所以","于是","不过","可见","同时","并且","而且","总之","其次","此外"
    }

def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    da = math.sqrt(sum(x*x for x in a)); db = math.sqrt(sum(x*x for x in b))
    if da == 0 or db == 0:
        return 0.0
    return sum(x*y for x, y in zip(a, b)) / (da * db)

def _cosine_tensor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """张量版本的余弦相似度计算"""
    if a.numel() == 0 or b.numel() == 0 or a.shape != b.shape:
        return torch.tensor(0.0, device=a.device)

    dot_product = torch.sum(a * b)
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)

    if norm_a == 0 or norm_b == 0:
        return torch.tensor(0.0, device=a.device)

    return dot_product / (norm_a * norm_b)

def split_keep_punct(text: str):
    return [m.group(0) for m in PAT.finditer(text)]

def build_high_entropy_blocks(
    tokens: List[str],
    entropies: List[float],
    *,
    # 阈值与平滑
    smooth_window: int = 5,
    seed_method: str = "mean_std",   # "mean_std" | "percentile"
    k_std: float = 1.0,              # τ = μ + k*σ
    percentile: float = 85.0,        # 或者用分位数阈值
    # 种子抑制与最小间隔
    local_max_radius: int = 1,
    min_seed_gap: int = 3,
    # 扩展规则
    drop_ratio: float = 0.45,        # 熵相对骤降停止阈
    floor_entropy: float = 0.0,      # 绝对下限
    max_block_len: int = 64,
    min_block_len: int = 3,
    # 语义边界
    stop_on_sentence_boundary: bool = True,
    include_connective_to_left: bool = True,
    # 合并与相邻拼接
    merge_overlap: bool = True,
    join_nearby_gap: int = 2,
    # 语义判断（可选）：传入 embedding 函数
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    query_text: Optional[str] = None,
    reference_texts: Optional[List[str]] = None,  # 多候选参考知识/答案
    sim_threshold_low: float = 0.35,              # 判为语义失效的低相似度阈
    reward_of_rollout: Optional[float] = None,    # e.g., 0/1 或标量回报
    advantage_of_rollout: Optional[float] = None # 若有优势估计，可用符号辅助判别
) -> List[Block]:
    """
    返回若干 Block，每个 Block 携带建议是否触发 SFT 的标记。
    """
    assert len(tokens) == len(entropies), "tokens 与 entropies 长度需一致"

    Hs = _gaussian_like_smooth(entropies, w=smooth_window)

    # 1) 阈值
    if seed_method == "mean_std":
        mu, sd = _mean_std(Hs)
        tau = mu + k_std * sd
    elif seed_method == "percentile":
        sorted_H = sorted(Hs)
        idx = int(len(sorted_H) * max(0.0, min(1.0, percentile / 100.0)))
        tau = sorted_H[idx] if sorted_H else 0.0
    else:
        raise ValueError("seed_method must be 'mean_std' or 'percentile'")

    # 2) 选种子：超过阈值且为局部极大；并做最小间隔约束
    seeds = []
    last_added = -10**9
    n = len(Hs)
    for i, h in enumerate(Hs):
        if h < tau:
            continue
        # 局部极大（或平台）
        left = max(0, i - local_max_radius)
        right = min(n, i + local_max_radius + 1)
        if all(h >= Hs[j] for j in range(left, right)):
            if i - last_added >= min_seed_gap:
                seeds.append(i); last_added = i
    # seeds = [i for i, h in enumerate(Hs) if h > entropy_threshold]
    # for seed in seeds:
    #     print(f'token: {tokens[seed]} \t entropy value: {Hs[seed]}')
    print(f'fraction of seeds: {len(seeds) / n:.2f}')

    colored_tokens = []
    for idx, token in enumerate(tokens):
        HUE_HIGH = 120  # 高概率色调（绿色）
        HUE_LOW = 0  # 低概率色调（红色）
        SATURATION = 0.7
        LIGHTNESS = 0.5
        prob_clamped = min(Hs[idx]/tau, 1.0)
        hue = HUE_LOW + (HUE_HIGH - HUE_LOW) * prob_clamped
        r, g, b = colorsys.hls_to_rgb(hue / 360, LIGHTNESS, SATURATION)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        if idx in seeds:
            colored_sub = f"\033[38;2;{r};{g};{b}m->{token}<-\033[0m"
        else:
            colored_sub = f"\033[38;2;{r};{g};{b}m{token}\033[0m"
        colored_tokens.append(colored_sub)
    print(''.join(colored_tokens))

    # 3) 从种子扩展
    blocks: List[Block] = []
    for s in seeds:
        peak = s
        seed_h = Hs[peak]
        # 左扩展：允许把连接词包含进来
        L = s
        while L - 1 >= 0 and (Hs[L-1] >= max(floor_entropy, drop_ratio * seed_h)):
            if stop_on_sentence_boundary and _is_sentence_boundary(tokens[L-1]):
                break
            L -= 1
            if s - L + 1 >= max_block_len:
                break
        if include_connective_to_left and L - 1 >= 0 and _is_connective(tokens[L-1]):
            L -= 1

        # 右扩展
        R = s + 1
        while R < n and (Hs[R] >= max(floor_entropy, drop_ratio * seed_h)):
            if stop_on_sentence_boundary and _is_sentence_boundary(tokens[R-1]):
                break
            R += 1
            if R - L >= max_block_len:
                break

        if R - L < min_block_len:
            continue

        block_H = Hs[L:R]
        b = Block(
            start=L,
            end=R,
            peak_idx=peak,
            mean_entropy=statistics.fmean(block_H),
            max_entropy=max(block_H),
            meta={"tau": tau, "seed_h": seed_h}
        )
        blocks.append(b)

    # 4) 合并/拼接
    if blocks:
        blocks.sort(key=lambda b: (b.start, b.end))
        merged = [blocks[0]]
        for b in blocks[1:]:
            prev = merged[-1]
            if merge_overlap and b.start <= prev.end:
                # overlap
                new = Block(
                    start=prev.start,
                    end=max(prev.end, b.end),
                    peak_idx=prev.peak_idx if Hs[prev.peak_idx] >= Hs[b.peak_idx] else b.peak_idx,
                    mean_entropy=statistics.fmean(Hs[prev.start:max(prev.end, b.end)]),
                    max_entropy=max(Hs[prev.start:max(prev.end, b.end)]),
                    meta={"merged": True}
                )
                merged[-1] = new
            elif b.start - merged[-1].end <= join_nearby_gap:
                # 邻近拼接
                jL, jR = merged[-1].start, b.end
                merged[-1] = Block(
                    start=jL,
                    end=jR,
                    peak_idx=merged[-1].peak_idx if Hs[merged[-1].peak_idx] >= Hs[b.peak_idx] else b.peak_idx,
                    mean_entropy=statistics.fmean(Hs[jL:jR]),
                    max_entropy=max(Hs[jL:jR]),
                    meta={"joined": True}
                )
            else:
                merged.append(b)
        blocks = merged

    # 5) 句子切分
    whole_sentence = "".join(tokens)
    tokenized_text = split_keep_punct(whole_sentence)
    spanned_blocks = []
    # for text in tokenized_text:
    #     print('text: ',text)
    # print('len of blocks: ', len( blocks))
    start_end_sets = set()
    for block in blocks:
        sentence = "".join(tokens[block.start:block.end])
        # print('incomplete block: ', sentence)
        for text in tokenized_text:
            if sentence in text:
                # print(f'"{sentence}" in "{text}"')
                start = block.start
                end = block.end
                # print(f'before: {start} - {end}')
                while start-1 >= 0 and tokens[start-1]+sentence in text:
                    sentence = tokens[start - 1] + sentence
                    start -= 1
                while end < len(tokens) and sentence+tokens[end] in text:
                    sentence = sentence + tokens[end]
                    end += 1
                if (start, end) in start_end_sets:
                    continue
                else:
                    start_end_sets.add((start, end))
                spanned_blocks.append(
                    Block(start=start, end=end, peak_idx=block.peak_idx, meta=block.meta,
                          mean_entropy=block.mean_entropy, max_entropy=block.max_entropy)
                )
                # print(f'after: {start} - {end}')
                break
    blocks = spanned_blocks
    # print('len of blocks: ', len(blocks))


    # 6) 语义级别的“探索失效”判定（可选）
    if embed_fn is not None:
        # 可选：预先对 query / refs 做一遍，detect_irrelevant 内部不依赖这些缓存也能跑
        # 这里只保留 b.sim_to_refs 以兼容旧字段；更详细信号写到 b.meta["irrelevant_signals"]
        for b in blocks:
            text = "".join(tokens[b.start:b.end])  # 若你的 tokenizer 需要空格，改为 " ".join

            # 级联语义无关检测（向量相似 + 词法 + 交叉编码器 + NLI + 困惑度下降），任一缺省亦可运行
            dec = detect_irrelevant(
                block_text=text,
                query_text=query_text,
                reference_texts=reference_texts or [],
                embed_fn=embed_fn,
                # cross_encoder_fn=cross_encoder_fn,  # 可为 None
                # nli_fn=nli_fn,  # 可为 None
                # ppl_fn=ppl_fn,  # 可为 None
                # bm25_fn=bm25_fn,  # 可为 None
                # batch_neg_refs=batch_neg_refs  # 同 batch 其他参考文本作负例，可为 None
            )

            # 兼容旧字段：用聚合余弦作为“与参考的相似度”
            b.sim_to_query = None
            b.sim_to_refs = dec["signals"].get("agg_cos", None)

            # 记录更丰富的诊断信号，便于日志/调参
            if b.meta is None:
                b.meta = {}
            b.meta["irrelevant_signals"] = dec["signals"]

            # 触发 SFT：无关 且 rollout 失败（或优势<0）
            rollout_failed = (reward_of_rollout is not None and reward_of_rollout <= 0.0) or \
                             (advantage_of_rollout is not None and advantage_of_rollout < 0.0)

            b.trigger_sft = bool(rollout_failed and dec["is_irrelevant"])
            b.reason = f"{dec['label']} | {dec['why']}"

    return spanned_blocks


def build_high_entropy_blocks_tensor(
        tokens: List[List[str]],  # 形状为 [bsz, length] 的字符串列表
        entropies: torch.Tensor,  # 形状为 [bsz, length] 的张量
        attention_mask: torch.Tensor,
        *,
        smooth_window: int = 5,
        seed_method: str = "mean_std",
        k_std: float = 1.0,
        percentile: float = 85.0,
        local_max_radius: int = 1,
        min_seed_gap: int = 3,
        drop_ratio: float = 0.45,
        floor_entropy: float = 0.0,
        max_block_len: int = 64,
        min_block_len: int = 3,
        stop_on_sentence_boundary: bool = True,
        include_connective_to_left: bool = True,
        merge_overlap: bool = True,
        join_nearby_gap: int = 2,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        query_text: Optional[str] = None,
        reference_texts: Optional[List[str]] = None,
        sim_threshold_low: float = 0.35,
        reward_of_rollout: Optional[float] = None,
        advantage_of_rollout: Optional[float] = None,
        verbose: bool = False  # 添加verbose参数控制输出
) -> List[List[Block]]:
    """
    批量处理版本的 build_high_entropy_blocks
    返回形状为 [bsz] 的 Block 列表
    """
    assert entropies.dim() == 2, "entropies 应该是二维张量"
    assert attention_mask.dim() == 2, "attention_mask 应该是二维张量"
    assert entropies.shape == attention_mask.shape, "entropies 和 attention_mask 形状应该相同"
    assert len(tokens) == entropies.size(0), "tokens 和 entropies 批次大小需一致"

    bsz, length = entropies.shape
    results = []

    # 对每个序列单独处理
    for i in range(bsz):
        # 获取当前序列的有效部分
        valid_indices = attention_mask[i].nonzero(as_tuple=True)[0]
        if valid_indices.numel() == 0:
            results.append([])
            continue
        # print('-----------')

        # print(''.join(tokens[i]))
        # print(f' length of tokens: {len(tokens[i])}, length of valid_indices: {valid_indices.numel()}')
        seq_tokens = [tokens[i][j] for j in valid_indices.cpu().numpy()]
        seq_entropies = entropies[i][valid_indices]
        assert len(seq_tokens) == seq_entropies.size(0), "tokens 和 entropies 形状应该一致"

        # 调用原始函数（需要稍作修改以支持张量）
        blocks = _process_single_sequence(
            seq_tokens, seq_entropies,
            smooth_window=smooth_window,
            seed_method=seed_method,
            k_std=k_std,
            percentile=percentile,
            local_max_radius=local_max_radius,
            min_seed_gap=min_seed_gap,
            drop_ratio=drop_ratio,
            floor_entropy=floor_entropy,
            max_block_len=max_block_len,
            min_block_len=min_block_len,
            stop_on_sentence_boundary=stop_on_sentence_boundary,
            include_connective_to_left=include_connective_to_left,
            merge_overlap=merge_overlap,
            join_nearby_gap=join_nearby_gap,
            embed_fn=embed_fn,
            query_text=query_text,
            reference_texts=reference_texts,
            sim_threshold_low=sim_threshold_low,
            reward_of_rollout=reward_of_rollout,
            advantage_of_rollout=advantage_of_rollout,
            verbose=verbose
        )
        # print('-----------')

        # 将块的位置映射回原始序列中的位置
        for block in blocks:
            block.start = valid_indices[block.start].item()
            block.end = valid_indices[block.end - 1].item() + 1 if block.end > 0 else 0
            block.peak_idx = valid_indices[block.peak_idx].item()

        results.append(blocks)

    return results


def _process_single_sequence(
        tokens: List[str],
        entropies: torch.Tensor,
        **kwargs
) -> List[Block]:
    """处理单个序列的辅助函数"""
    # 这里实现原始 build_high_entropy_blocks 的逻辑，但使用张量操作
    # 由于代码较长，以下只展示关键修改部分

    # 提取参数
    verbose = kwargs.get('verbose', True)
    smooth_window = kwargs.get('smooth_window', 5)

    # 平滑处理
    Hs = _gaussian_like_smooth_tensor(entropies, w=smooth_window)

    # 计算阈值
    if kwargs.get('seed_method') == "mean_std":
        # 创建有效值的mask（全部为True，因为我们已经过滤了无效值）
        mask = torch.ones_like(Hs, dtype=torch.bool)
        mu, sd = _mean_std_tensor(Hs, mask)
        tau = mu + kwargs.get('k_std', 1.0) * sd
    elif kwargs.get('seed_method') == "percentile":
        # 计算分位数
        sorted_H, _ = torch.sort(Hs)
        idx = int(len(sorted_H) * max(0.0, min(1.0, kwargs.get('percentile', 85.0) / 100.0)))
        tau = sorted_H[idx] if len(sorted_H) > 0 else torch.tensor(0.0)
    else:
        raise ValueError("seed_method must be 'mean_std' or 'percentile'")

    # 选择种子点
    n = len(Hs)
    seeds = []
    last_added = -10 ** 9

    # 寻找局部最大值
    for i in range(n):
        h = Hs[i]
        if h < tau:
            continue

        # 检查是否是局部最大值
        left = max(0, i - kwargs.get('local_max_radius', 1))
        right = min(n, i + kwargs.get('local_max_radius', 1) + 1)

        is_local_max = True
        for j in range(left, right):
            if Hs[j] > h:  # 注意这里使用了严格大于，与原代码稍有不同
                is_local_max = False
                break

        if is_local_max and (i - last_added) >= kwargs.get('min_seed_gap', 3):
            seeds.append(i)
            last_added = i

    if verbose:
        print(f'fraction of seeds: {len(seeds) / n:.2f}')
        # 彩色输出代码需要修改以适应张量，这里省略

    # 从种子点扩展（这里需要更多修改以实现向量化，但为了清晰度，保持循环）
    blocks = []
    for s in seeds:
        # 扩展逻辑与原函数类似，但使用张量操作
        peak = s
        seed_h = Hs[peak]

        # 左扩展
        L = s
        while L - 1 >= 0 and (
                Hs[L - 1] >= max(kwargs.get('floor_entropy', 0.0), kwargs.get('drop_ratio', 0.45) * seed_h)):
            if kwargs.get('stop_on_sentence_boundary', True) and _is_sentence_boundary(tokens[L - 1]):
                break
            L -= 1
            if s - L + 1 >= kwargs.get('max_block_len', 64):
                break

        if kwargs.get('include_connective_to_left', True) and L - 1 >= 0 and _is_connective(tokens[L - 1]):
            L -= 1

        # 右扩展
        R = s + 1
        while R < n and (Hs[R] >= max(kwargs.get('floor_entropy', 0.0), kwargs.get('drop_ratio', 0.45) * seed_h)):
            if kwargs.get('stop_on_sentence_boundary', True) and _is_sentence_boundary(tokens[R - 1]):
                break
            R += 1
            if R - L >= kwargs.get('max_block_len', 64):
                break

        if R - L < kwargs.get('min_block_len', 3):
            continue

        block_H = Hs[L:R]
        b = Block(
            start=L,
            end=R,
            peak_idx=peak,
            mean_entropy=torch.mean(block_H).item(),
            max_entropy=torch.max(block_H).item(),
            meta={"tau": tau.item(), "seed_h": seed_h.item()}
        )
        blocks.append(b)

    # 合并/拼接块（逻辑与原函数类似，但使用张量操作）
    if blocks:
        blocks.sort(key=lambda b: (b.start, b.end))
        merged = [blocks[0]]
        for b in blocks[1:]:
            prev = merged[-1]
            if kwargs.get('merge_overlap', True) and b.start <= prev.end:
                # 处理重叠
                new_start = prev.start
                new_end = max(prev.end, b.end)
                new_Hs = Hs[new_start:new_end]

                merged[-1] = Block(
                    start=new_start,
                    end=new_end,
                    peak_idx=prev.peak_idx if Hs[prev.peak_idx] >= Hs[b.peak_idx] else b.peak_idx,
                    mean_entropy=torch.mean(new_Hs).item(),
                    max_entropy=torch.max(new_Hs).item(),
                    meta={"merged": True}
                )
            elif b.start - merged[-1].end <= kwargs.get('join_nearby_gap', 2):
                # 拼接邻近块
                jL, jR = merged[-1].start, b.end
                joined_Hs = Hs[jL:jR]

                merged[-1] = Block(
                    start=jL,
                    end=jR,
                    peak_idx=merged[-1].peak_idx if Hs[merged[-1].peak_idx] >= Hs[b.peak_idx] else b.peak_idx,
                    mean_entropy=torch.mean(joined_Hs).item(),
                    max_entropy=torch.max(joined_Hs).item(),
                    meta={"joined": True}
                )
            else:
                merged.append(b)
        blocks = merged

    # 句子切分和语义判断部分需要更多工作来适配张量
    # 这里省略了这些部分的实现，但逻辑应与原函数类似
    whole_sentence = "".join(tokens)
    tokenized_text = split_keep_punct(whole_sentence)
    spanned_blocks = []
    # for text in tokenized_text:
    #     print('text: ',text)
    # print('len of blocks: ', len( blocks))
    start_end_sets = set()
    for block in blocks:
        sentence = "".join(tokens[block.start:block.end])
        # print('incomplete block: ', sentence)
        for text in tokenized_text:
            if sentence in text:
                # print(f'"{sentence}" in "{text}"')
                start = block.start
                end = block.end
                # print(f'before: {start} - {end}')
                while start - 1 >= 0 and (tokens[start - 1] + sentence).strip() in text:
                    sentence = tokens[start - 1] + sentence
                    start -= 1
                while end < len(tokens) and (sentence + tokens[end]).strip() in text:
                    sentence = sentence + tokens[end]
                    end += 1
                if (start, end) in start_end_sets:
                    continue
                else:
                    start_end_sets.add((start, end))
                spanned_blocks.append(
                    Block(start=start, end=end, peak_idx=block.peak_idx, meta=block.meta,
                          mean_entropy=block.mean_entropy, max_entropy=block.max_entropy, text=sentence.strip())
                )
                # print(f'after: {start} - {end}\t{sentence}')
                if sentence.strip() != text.strip():
                    continue
                else:
                    break
    blocks = spanned_blocks

    return blocks

# ------- 小示例 -------
if __name__ == "__main__":
    toks = ["王","熙","凤","逼","死","尤","二","姐","，","可","见","她","手","段","残","酷","。","然","而","在","…"]
    H =   [0.2,0.25,0.28,0.35,0.92,0.88,0.8,0.7,0.1,0.22,0.24,0.26,0.3,0.33,0.31,0.29,0.1,0.4,0.42,0.2,0.18]

    # 假设没有嵌入函数，就先做块划分
    _blocks = build_high_entropy_blocks(
        toks, H,
        smooth_window=3, seed_method="mean_std", k_std=0.8,
        drop_ratio=0.5, max_block_len=16, min_block_len=3,
        stop_on_sentence_boundary=True
    )
    # for b in blocks:
        # print(b.start, b.end, "".join(toks[b.start:b.end]), b.mean_entropy, b.trigger_sft, b.reason)
