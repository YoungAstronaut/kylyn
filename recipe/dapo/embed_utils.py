import torch
import numpy as np

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

TASK = ("Given a mathematical or commonsense reasoning step, "
        "retrieve the most semantically equivalent step from the reference solution.")

TASK_PREFIX = (
    "Given a cumulative reasoning prefix, evaluate whether its semantic similarity "
    "to the reference solution drops compared to the previous prefix;"
    "focus on mathematical equivalence, shared variables/constants, and transformations, "
    "and ignore phrasing or order differences."
)

# TASK_PREFIX = (
#     "Given a cumulative reasoning prefix, evaluate whether its semantic similarity "
#     "to the reference solution drops compared to the previous prefix."
# )

def build_embed_inputs(gen_steps: list[str], ref_steps: list[str], task_description: str) -> tuple[list[str], int, int]:
    """
    返回:
        input_texts:  [q0,...,qm-1, r0,...,rk-1]  (带指令的query + 无指令的reference)
        m, k        : 方便后面切片
    """
    queries = [get_detailed_instruct(task_description, s) for s in gen_steps]
    input_texts = queries + ref_steps          # reference 不需要指令
    return input_texts, len(gen_steps), len(ref_steps)

def find_first_descent_basic(data: list[float]) -> int:
    """
    找到第一个简单下降沿的位置（当前值 < 前一个值）

    返回: 下降沿的起始索引（即较高值的索引），如果没有返回-1
    """
    if len(data) < 2:
        return -1

    for i in range(1, len(data)):
        if data[i] < data[i - 1]:
            return i  # 返回下降开始的索引

    return -1

def find_first_descent_with_threshold(data: list[float], threshold: float = 0.01) -> int:
    """
    找到第一个显著下降沿的位置

    参数:
        data: float列表
        threshold: 最小下降幅度阈值（前一个值 - 当前值）

    返回: 下降沿的起始索引，如果没有返回-1
    """
    if len(data) < 2:
        return -1

    for i in range(1, len(data)):
        if data[i - 1] - data[i] > threshold:
            return i

    return -1

def find_first_descent_point(data: list[float], drop_threshold: float = 0.005) -> int:
    """
    找到下降沿中第一个与最高点差值≥阈值的点

    参数:
        data: float列表
        drop_threshold: 与最高点的最小下降幅度阈值

    返回: 第一个满足条件的下降点索引，如果没有返回-1
    """
    if len(data) < 2:
        return -1

    i = 0
    while i < len(data) - 1:
        # 阶段1：寻找峰值（上升沿终点）
        while i < len(data) - 1 and data[i + 1] >= data[i]:
            i += 1

        if i >= len(data) - 1:  # 已到末尾，无下降
            break

        peak_value = data[i]  # 记录最高点的值

        # 阶段2：从峰值后开始检查，找第一个满足阈值的点
        j = i + 1
        while j < len(data):
            # 关键点：立即检查是否满足条件
            if peak_value - data[j] >= drop_threshold:
                return j  # ✅ 找到第一个满足条件的点，立即返回

            # 如果开始上升，当前下降沿结束
            if data[j] > data[j - 1]:
                break

            j += 1

        # 当前下降沿未满足阈值，从下一位置继续搜索
        i = j

    return -1

def argmin(lst: list[float]):
    return lst.index(min(lst))

@torch.no_grad()
def step_similarity_matrix(gen_steps: list[str], ref_steps: list[str], task_description: str, model) -> torch.Tensor:
    """
    返回: Float[Tensor, "m k"]   语义相似度矩阵（cosine）
    """
    input_texts, m, k = build_embed_inputs(gen_steps, ref_steps, task_description)
    for text in input_texts:
        print([text])

    outputs = model.embed(input_texts)
    embs = torch.tensor([o.outputs.embedding for o in outputs])  # shape (m+k, dim)
    q_embs, r_embs = embs[:m], embs[m:]                  # 分离 query / reference
    q_embs = torch.nn.functional.normalize(q_embs, p=2, dim=1)
    r_embs = torch.nn.functional.normalize(r_embs, p=2, dim=1)
    scores = q_embs @ r_embs.T                           # (m, k)
    return scores

def locate_bad_steps(gen_steps: list[str], ref_steps: list[str],
                     threshold: float = 0.70) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sim_mat = step_similarity_matrix(gen_steps, ref_steps, TASK)  # (m, k)
    print(sim_mat)
    best_sim, best_ref_idx = sim_mat.max(dim=1)           # (m,)
    bad_mask = best_sim < threshold
    return bad_mask, best_ref_idx, best_sim

def locate_bad_prefixes(gen_steps: list[str], ref_steps: list[str], model,
                     threshold: float = 0.70) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prefixes = [''.join(gen_steps[:i+1]) for i in range(len(gen_steps))]
    for prefix in prefixes:
        print([prefix])
    sim_mat = step_similarity_matrix(prefixes, ref_steps, TASK_PREFIX, model)  # (m, k)
    best_sim, best_ref_idx = sim_mat.max(dim=1)           # (m,)
    bad_mask = best_sim < threshold
    return bad_mask, best_ref_idx, best_sim

def locate_bad_prefix_steps(
        gen_steps: list[str],
        ref_steps: list[str],
        threshold: float = 0.75
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    逐前缀判断相似度是否跌落且低于阈值
    返回:
        bad_mask: Bool[m]  True→该step导致跌落
        best_ref_idx: Long[m] 每一步前缀最像的ref_step下标
        best_sim: Float[m]  对应的最大相似度
    """
    m, k = len(gen_steps), len(ref_steps)
    if m == 0 or k == 0:
        return (torch.zeros(m, dtype=torch.bool),
                torch.zeros(m, dtype=torch.long),
                torch.zeros(m))

    # 1. 构造前缀序列
    prefix_steps = [' '.join(gen_steps[:i+1]) for i in range(m)]   # 0~i 拼接

    # 2. 编码前缀 + 参考
    prefix_steps = [get_detailed_instruct(TASK_PREFIX, p) for p in prefix_steps]
    sim_mat = step_similarity_matrix(prefix_steps, ref_steps, TASK_PREFIX)  # (m, k)

    best_sim, best_ref_idx = sim_mat.max(dim=1)                # (m,)

    # 3. 判断跌落
    bad_mask = torch.zeros(m, dtype=torch.bool)
    for i in range(1, m):
        if best_sim[i] < threshold and best_sim[i] < best_sim[i-1]:
            bad_mask[i] = True   # 把“导致跌落”的step标出来

    return bad_mask, best_ref_idx, best_sim


@torch.no_grad()
def balance_embeddings_batch(
        gen_steps_list: list[list[str]],
        ref_steps_list: list[list[str]],
        task_description: str,
        max_batch_size: int = 1,
        max_total_tokens: int = 16384,  # 新增：单批次最大token数限制
        tokenizer=None,  # 可选：如果提供则精确计算token数
        strategy: str = "naive"
) -> list[list[dict]]:
    """
    针对前缀场景优化的批量相似度计算（考虑累积长度增长）

    返回示例：
    [
        [ # batch 1
            { # 对应一个回复的若干step
                texts: [],
                pair_idx: 0,
                'start_idx': 0,
                'm': 8,
                'k': 1,
                'token_cost': total_tokens,
                'combinations': m * k,
            },
            {
                texts: [],
                pair_idx: 1,
                ...
            },
            ...
        ],
        ...
        [
            ...
        ]
    ]
    """
    assert len(gen_steps_list) == len(ref_steps_list), "输入列表长度必须一致"

    n_pairs = len(gen_steps_list)
    if n_pairs == 0:
        return []

    # all_input_texts = []
    pair_metadata = []
    token_costs = []  # 记录每个pair的预估token成本

    for idx, (gen_steps, ref_steps) in enumerate(zip(gen_steps_list, ref_steps_list)):
        if len(gen_steps) == 0:
            pair_metadata.append({'pair_idx': -1})
            continue
        # 构建输入文本
        assert len(gen_steps) > 0, f'm: {len(gen_steps)}, k: {len(ref_steps)}'
        # print(f'm: {len(gen_steps)}, k: {len(ref_steps)}')
        texts, m, k = build_embed_inputs(gen_steps, ref_steps, task_description)

        # 估算token数（累积前缀特性：越往后越长）
        if tokenizer is not None:
            # 精确计算
            total_tokens = sum(len(tokenizer.encode(t)) for t in texts)
        else:
            # 近似估算：字符数/4，并给前缀列表乘一个增长系数
            base_chars = sum(len(t) for t in texts)
            # 前缀增长惩罚因子（因为前缀越来越长，缓存效率低）
            prefix_penalty = 1 + 0.5 * np.log1p(len(gen_steps))
            total_tokens = int(base_chars / 4 * prefix_penalty)

        # all_input_texts.extend(texts)
        pair_metadata.append({
            'pair_idx': idx,
            # 'start_idx': len(all_input_texts) - len(texts),
            'm': m,
            'k': k,
            'token_cost': total_tokens,
            'combinations': m * k,
            'texts': texts,
        })
        token_costs.append(total_tokens)

    if strategy == "token_balanced":
        # 按token成本升序排序（优先处理小成本任务）
        pair_metadata.sort(key=lambda x: x['token_cost'])

        # 动态批次构建：尽可能填满token预算
        batches = []
        current_batch = []
        current_token_sum = 0

        for meta in pair_metadata:
            # 如果加入当前pair会超出限制，先提交批次
            if current_batch and current_token_sum + meta['token_cost'] > max_total_tokens:
                batches.append(current_batch)
                print('tokens of this batch: ', current_token_sum)
                current_batch = []
                current_token_sum = 0

            # 对于超大pair，单独成批
            if meta['token_cost'] > max_total_tokens:
                batches.append([meta])
                print('tokens of this batch: ', meta['token_cost'])
                continue

            current_batch.append(meta)
            current_token_sum += meta['token_cost']

        if current_batch:
            batches.append(current_batch)
            print('tokens of this batch: ', current_token_sum)

    elif strategy == "size_balanced":
        # 按组合数排序的传统策略
        pair_metadata.sort(key=lambda x: x['combinations'])
        batches = [pair_metadata[i:i + max_batch_size]
                   for i in range(0, len(pair_metadata), max_batch_size)]
    else:
        # naive策略：保持原始顺序
        batches = [pair_metadata[i:i + max_batch_size]
                   for i in range(0, len(pair_metadata), max_batch_size)]

    return batches