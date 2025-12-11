import os
import json
import re
import time
import random
import requests
import concurrent.futures as cf
from collections import Counter
from typing import Optional, List

# ========== 基本配置 ==========
MODEL = "deepseek-chat"
API_KEY = os.getenv("PROBEX_API_KEY", "sk-REPLACE_ME")  # 建议用环境变量
API_URL = "https://api.probex.top/v1/chat/completions"

# 最大并发线程数：根据你接口限流和机器带宽调节，常见 4~16 之间
MAX_WORKERS = 8
# 针对 429/5xx 的最大重试次数
MAX_RETRIES = 3
# 每次重试的基础退避秒数（会乘以 2^attempt，并加少量随机抖动）
BACKOFF_BASE = 1.0

# ========== 全局会话：复用连接，显著降低开销 ==========
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_WORKERS * 2,
                                        pool_maxsize=MAX_WORKERS * 2)
session.mount("https://", adapter)
session.headers.update({
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
})

def parse_label_from_text(text: str) -> Optional[int]:
    """
    从模型输出中解析标签（1/2/3/4）。尽量稳健：优先匹配独立数字，其次做回退检查。
    """
    if not text:
        return None
    # 优先匹配独立数字（避免把 10 误判为 1）
    m = re.search(r"\b([1-4])\b", text)
    if m:
        return int(m.group(1))
    # 回退：从后往前找任意包含 1~4 的行（与原逻辑一致）
    for line in reversed(text.splitlines()):
        for d in ("1", "2", "3", "4"):
            if d in line:
                return int(d)
    return None

def send_message(message: str) -> Optional[str]:
    """
    发送请求到接口，带重试与指数退避。
    返回模型 content（字符串），失败则返回 None。
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": message}],
        "stream": False
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = session.post(API_URL, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                # 常见限流/服务端错误：尝试重试
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                    sleep_s = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.3)
                    time.sleep(sleep_s)
                    continue
                # 其他错误直接返回 None，并打印错误
                print(f"[ERROR] HTTP {resp.status_code}: {resp.text[:300]}")
                return None
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES:
                sleep_s = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.3)
                time.sleep(sleep_s)
                continue
            print(f"[ERROR] Request failed after retries: {e}")
            return None

def build_message_from_file(file_path: str) -> Optional[str]:
    """
    读取 JSON 文件并构造 prompt。
    失败返回 None。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        prefix_answer = data['prefix_answer'].split(
            'Your task is to understand a given standard problem solving process'
            ' of a given question, then finish an incomplete reasoning process. '
            'The question is :\n'
        )[1]
        question = prefix_answer.split('The standard solving process is as followings:')[0].strip()
        prefix_answer = prefix_answer.split('The standard solving process is as followings:')[1]
        standard_process = prefix_answer.split('User: **Finish the following incomplete answer**:')[0].strip()
        incomplete_answer = prefix_answer.split('User: **Finish the following incomplete answer**:')[1].strip()

        answer_before = data['answer_before']
        answer_after = data['answer_after']
        step_before = answer_before.split(incomplete_answer)[1]
        step_after = answer_after.split(incomplete_answer)[1]

        message = (
            f"我需要你认真地帮我解决以下问题。我现在有一个数学推理问题：\n{question} \n"
            f"这个问题的答案是：{standard_process}\n"
            f"我现在让一个功能相对弱的模型回答了这个问题，模型先回答了一部分如下：\n{incomplete_answer} \n"
            f"现在有两个可能紧接着以上不完整的解答的步骤，请你判断以下两个步骤是否能导致模型回答的答案正确，"
            f"请用数字1表示只有第一个步骤正确，数字2表示只有第二个步骤正确，数字3表示两个步骤都正确，"
            f"数字4表示两个步骤都不正确：\n第一个步骤：\"{step_before}\" \n第二个步骤：\"{step_after}\" "
        )
        return message
    except Exception as e:
        print(f"[ERROR] Failed to build message from {file_path}: {e}")
        return None

def judge_answer(file_path: str) -> Optional[int]:
    """
    单文件评测（线程工作函数）：构造消息 -> 调用接口 -> 解析标签。
    """
    message = build_message_from_file(file_path)
    if message is None:
        return None
    result_text = send_message(message)
    if result_text is None:
        print(f"[WARN] No result for {file_path}")
        return None
    label = parse_label_from_text(result_text)
    if label not in (1, 2, 3, 4):
        print(f"[WARN] Unrecognized label for {file_path}: {result_text[:200]}")
        return None
    return label

def list_target_files(path_prefix: str, limit: Optional[int] = None) -> List[str]:
    """
    枚举目标目录下的 JSON 文件；可限制数量。
    """
    try:
        names = os.listdir(path_prefix)
    except FileNotFoundError:
        print(f"[ERROR] Directory not found: {path_prefix}")
        return []
    files = [os.path.join(path_prefix, n) for n in names if n.lower().endswith(".json")]
    if limit is not None and limit > 0:
        return files[:limit]
    return files

def main():
    path_prefix = "parsed_coef_0.1/1/"
    # 如果你只想跑前 N 个文件，在这里改，例如 N=10；不限制就传 None
    files = list_target_files(path_prefix, limit=None)

    if not files:
        print("[INFO] No files to process.")
        return

    counts = Counter()
    results = {}

    print(f"[INFO] Start concurrent processing: {len(files)} files, max_workers={MAX_WORKERS}")

    # 线程池并发执行
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(judge_answer, fp): fp for fp in files}

        for future in cf.as_completed(future_to_file):
            fp = future_to_file[future]
            try:
                label = future.result()
                results[fp] = label
                if label in (1, 2, 3, 4):
                    counts[label] += 1
                    print(f"[OK] {fp} -> {label}")
                else:
                    print(f"[SKIP] {fp} -> None/Invalid")
            except Exception as e:
                print(f"[ERROR] {fp} raised: {e}")

    # 汇总
    print("\n========== Summary ==========")
    for k in (1, 2, 3, 4):
        print(f"Label {k}: {counts[k]}")
    print(f"Total valid: {sum(counts.values())} / {len(files)}")

if __name__ == "__main__":
    main()
