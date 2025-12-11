import json
from typing import List, Optional, Dict, Any
from openai import OpenAI

# MODEL_USED="Kimi-K2-Instruct"
MODEL_USED="gpt-4o"
# ------------------ 核心：判定单个候选步骤 ------------------
def judge_candidate_step_chat(
    problem: str,
    prefix_steps: List[str],
    candidate_step: str,
    reference_answer: Optional[str] = None,
    *,
    model: str = MODEL_USED,
    client: Optional[OpenAI] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    用 Chat Completions + Function Calling 判定候选步骤是否正确。
    返回 JSON(dict): {
        "is_correct": bool,
        "error_type": "算术/代数错误" | "不充分推断" | "与前文矛盾" | "符号/定义误用" | "跳步且关键缺失" | "无",
        "brief_reason": str,
        "minimal_fix": str,
        "confidence": float
    }
    """
    client = client or OpenAI()

    # —— 函数模式(工具)的参数 schema：让模型“调用函数”并只产出结构化参数
    tools = [{
        "type": "function",
        "function": {
            "name": "step_verdict",
            "description": "Provide a structured verdict for the candidate next step.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "is_correct": {"type": "boolean"},
                    # "error_type": {
                    #     "type": "string",
                    #     "enum": [
                    #         "arithmetic/algebraic error",
                    #         "insufficient inference",
                    #         "contradiction with previous steps",
                    #         "misuse of symbols/definitions",
                    #         "skipped key steps",
                    #         "none"
                    #     ]
                    # },
                    "brief_reason": {"type": "string", "maxLength": 100},
                    "minimal_fix": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["is_correct", "brief_reason", "minimal_fix", "confidence"]
            }
        }
    }]

    # system_msg = (
    #     "你是一名严格的数学步骤审稿人。仅判断“候选下一步”是否能在逻辑与数学上"
    #     "由题目与前缀步骤推出；若不成立，指出最小问题与最小修正。"
    #     "请务必通过调用函数 step_verdict 输出结构化结果，不要长篇推导。"
    # )
    system_msg = (
        "You are a strict reviewer of mathematical solution steps. "
        "Only decide whether the 'candidate next step' can be logically and mathematically "
        "derived from the problem statement and the preceding steps; if not, point out the "
        "minimal issue and the minimal correction. "
        "You must output a structured result by calling the function step_verdict and you "
        "should not provide long derivations."
    )

    prefix_block = "\n".join(f"{i+1}) {s}" for i, s in enumerate(prefix_steps)) or "(空)"
    reference_block = reference_answer or "(未提供)"
    user_msg = (
        f"[Problem]\n{problem}\n\n"
        f"[Verified Prefix Steps]\n{prefix_block}\n\n"
        f"[Candidate Step]\n{candidate_step}\n\n"
        # f"[Reference Answer]\n{reference_block}\n\n"
        "[Task]\n"
        "Determine whether the Candidate Step is correct:\n"
        "- If correct: is_correct=true, error_type=\"none\", minimal_fix is an empty string "
        "or \"none needed\".\n"
        "- If incorrect: is_correct=false, choose the most appropriate error_type; "
        "if the candidate step is just logical connection words, consider the step correct;"
        "brief_reason should be at most 100 characters; minimal_fix should provide the "
        "minimal correction in a single sentence.\n"
        "Return the result via a call to the function step_verdict."
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "step_verdict"}},  # 强制函数调用
    )

    choice = resp.choices[0]
    # 期望 finish_reason == "tool_calls"
    tool_calls = getattr(choice.message, "tool_calls", None)

    if tool_calls:
        args_str = tool_calls[0].function.arguments
        try:
            result = json.loads(args_str)
            result['user_msg'] = user_msg
            print(result)
            return result
        except json.JSONDecodeError:
            # 极端情况下模型参数带尾随文本，尽量截断到第一个完整对象
            first_brace = args_str.find("{")
            last_brace = args_str.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                # return json.loads(args_str[first_brace:last_brace + 1])
                return {'failed': f'{args_str[first_brace:last_brace + 1]}'}
            raise RuntimeError(f"函数参数无法解析为 JSON: {args_str}")

    # 兜底：模型没有按函数调用返回，尝试直接把文本解析成 JSON（不推荐，但提高健壮性）
    txt = choice.message.content or ""
    try:
        return json.loads(txt)
    except Exception:
        raise RuntimeError(
            "模型未进行函数调用且输出无法解析为 JSON。"
            f"\n原始输出：{txt}"
        )


# ------------------ 辅助：遍历定位首错 ------------------
def localize_first_error_chat(
    problem: str,
    steps: List[str],
    reference_answer: Optional[str] = None,
    model: str = MODEL_USED,
    client: Optional[OpenAI] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    从前到后依次判定，返回首错索引与结论。
    返回:
    {
        "k": int | None,           # 首错步骤下标(1-based)，若未发现则为 None
        "verdict": Dict[str, Any]  # 与 judge_candidate_step_chat 的返回一致
    }
    """
    prefix: List[str] = []
    for i, step in enumerate(steps, start=1):
        print(f'step {i}: {step}')
        print(f'model: {model}')
        try_times = 0
        verdict = None
        while try_times < 3:
            verdict = judge_candidate_step_chat(
                problem=problem,
                prefix_steps=prefix,
                candidate_step=step,
                reference_answer=reference_answer,
                model=model,
                client=client,
                temperature=temperature,
            )
            if 'failed' not in verdict.keys():
                print(verdict)
                break
            else:
                try_times += 1
                print(f"try times: {try_times}")
        if not verdict.get("is_correct", False):
            return {"k": i, "verdict": verdict, 'steps': steps}
        prefix.append(step)

    # 到这说明每步都判为正确；如果给了标准答案但最终不一致，按需回溯（可自定义）
    return {"k": None, "verdict": {"is_correct": True, "error_type": "无",
                                   "brief_reason": "未定位到首错", "minimal_fix": "", "confidence": 0.8, "steps": steps}}