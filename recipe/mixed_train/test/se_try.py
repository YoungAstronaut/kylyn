# save as run_vllm_batch.py
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Dict, Any

from openai import OpenAI
from vllm import LLM, SamplingParams

from recipe.mixed_train.semantic_blocks import convert_ids_to_text_splits, split_into_blocks, text_to_pieces
from recipe.mixed_train.step_localization import judge_candidate_step_chat
from verl.utils import hf_tokenizer


# def construct_explain_prompt(question: str, standard_answer: str, answer_prefix: str):
#     chat = [ # 先给定问题和标准答案，再给定前缀要求模型补全
#         {
#             "content": f"Your task is to understand a given standard problem solving process of a given question, "
#                        f"then finish an incomplete reasoning process. The question is :\n{question}\nThe standard "
#                        f"solving process is as followings: \n\"\n{standard_answer}\n\"\n",
#             "role": "system"
#         },
#         {
#             "content": f"**Finish the following incomplete answer**: \n{answer_prefix}",
#             "role": "user"
#         }
#     ]
#     # chat = [  # 只告诉模型错了，相当于重写
#     #     {
#     #         "content": f"Your task is to finish a incomplete reasoning process. The question is :\n{question}\n Think "
#     #                    f"about the problem first.",
#     #         "role": "system"
#     #     },
#     #     {
#     #         "content": f"**Finish the following incomplete answer**: \n{answer_prefix}",
#     #         "role": "user"
#     #     }
#     # ]
#     result = chat[0]["content"]+" User: "+chat[1]["content"]
#     # k = result.replace('\n', '&&')
#     # print(f' explain prompt: {k}')
#     return result

def construct_explain_prompt(question: str, standard_answer: str, answer_prefix: str):
    system_content = (
        "You are an expert step-by-step problem solving assistant.\n"
        "You will be given a question, its standard solution, and an incomplete reasoning prefix.\n"
        "Your job is to continue the incomplete reasoning so it becomes a correct and complete solution.\n"
        "Follow the standard solution when necessary, but keep the existing prefix as much as possible.\n"
        "Only output the continuation, do not repeat the prefix."
    )

    user_part = (
        f"[Question]\n{question}\n\n"
        f"[Standard solution]\n{standard_answer}\n\n"
        f"[Incomplete answer prefix]\n{answer_prefix}\n\n"
        "Continue the reasoning **from the last line of the incomplete answer prefix**.\n"
        "Remember: only output the continuation, not the prefix itself:\n"
    )

    result = system_content + "\nUser:\n" + user_part
    return result


def read_prompts(path: str) -> dict:
    prompts = []
    problems = []
    answer_prefixs = []
    standard_answers = []
    incorrect_steps = []
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        answer_prefix = item['answer_prefix']
        standard_answer = item['standard_answer']
        problem = item['problem']
        incorrect_step = item['step_incorrect']['step']
        prompts.append(construct_explain_prompt(problem, standard_answer, answer_prefix))
        answer_prefixs.append(answer_prefix)
        problems.append(problem)
        standard_answers.append(standard_answer)
        incorrect_steps.append(incorrect_step)
    return {
        "prompts": prompts,
        "answer_prefixs": answer_prefixs,
        "problems": problems,
        "standard_answers": standard_answers,
        "incorrect_steps": incorrect_steps
    }

def eval_step(item, llm_client):
    """在线程中调用 judge_candidate_step_chat 的封装函数。"""
    try:
        if len(item["steps"]) == 0:
            return None
        verdict = judge_candidate_step_chat(
            problem=item["problem"],
            prefix_steps=[item['answer_prefix']],
            candidate_step=item["steps"][0],
            reference_answer=item["standard_answer"],
            client=llm_client,
        )
        verdict["step"] = item["steps"][0]
        print(verdict)
        item["verdict"] = verdict
        return item["index"], item
    except AttributeError as e:
        print("WARNING: ", e)
        return None

def main():
    p = argparse.ArgumentParser("Batch inference with vLLM")
    p.add_argument('--rollout_times', type=int, default=4)
    p.add_argument("--model", type=str, required=True,
                   help="HF / model scope 名称或本地路径，如: Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--input", type=str, required=True,
                   help="输入文件：.txt(一行一个) 或 .jsonl(字段名 prompt)")
    p.add_argument("--output", type=str, default="outputs.json",
                   help="输出 JSONL 路径")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--frequency-penalty", type=float, default=0.0)
    p.add_argument("--stop", type=str, nargs="*", default=None,
                   help="可选：一个或多个 stop 词")
    p.add_argument("--dtype", type=str, default="auto",
                   help="auto / float16 / bfloat16 / float32")
    p.add_argument("--tp", type=int, default=1,
                   help="tensor_parallel_size，多卡时 >=2")
    p.add_argument("--gpu-mem", type=float, default=0.95,
                   help="gpu_memory_utilization (0~1)")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="某些模型需要 True")
    args = p.parse_args()

    llm_client = OpenAI(base_url="https://api.vectorengine.ai/v1",
                        api_key="sk-PqqzpkgeXymtXSLepUSnK9XAuluuyEaRITaXjugJgm22fdwj")

    data = read_prompts(args.input)
    prompts = data["prompts"]
    answer_prefixs = data["answer_prefixs"]
    problems = data["problems"]
    standard_answers = data["standard_answers"]
    incorrect_steps = data["incorrect_steps"]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"#prompts={len(prompts)}  tp={args.tp}")

    tokenizer = hf_tokenizer(args.model)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=args.trust_remote_code,
    )

    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=args.stop,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
    )

    ratios = [0] * len(prompts)
    all_results = []
    batch = []
    for prompt in prompts:
        batch += [prompt] * args.rollout_times
    results = llm.generate(batch, sampling)
    generated = []
    for i in range(len(prompts)):
        answer_prefix = answer_prefixs[i]
        problem = problems[i]
        standard_answer = standard_answers[i]
        incorrect_step = incorrect_steps[i]
        # results 与输入顺序一致
        for out in results[i * args.rollout_times: (i + 1) * args.rollout_times]:
            text = out.outputs[0].text if out.outputs else ""
            # print(f' generated text: {text}')
            # token_ids = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
            # print(out.outputs[0].token_ids)
            # tokens = convert_ids_to_text_splits(token_ids, tokenizer)
            tokens = text_to_pieces(text, tokenizer)
            try:
                _, complete_answer_splits = split_into_blocks(text, tokens, 192)
                generated.append({
                    "prompt": out.prompt,
                    "output": text,
                    "finish_reason": out.outputs[0].finish_reason if out.outputs else None,
                    "tokens_num": len(out.outputs[0].token_ids) if out.outputs else 0,
                    "steps": complete_answer_splits,
                    "problem": problem,
                    "answer_prefix": answer_prefix,
                    "standard_answer": standard_answer,
                    "index": i,
                    "incorrect_step": incorrect_step
                })
            except AttributeError:
                continue

    corrected_num = 0
    # === 这里开始：多线程调用 judge_candidate_step_chat ===
    max_workers = 128
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                eval_step,
                item,
                llm_client,
            )
            for item in generated
        ]

        for future in as_completed(futures):
            index, item = future.result()
            if item is None:
                continue
            if item["verdict"].get("is_correct", False):
                corrected_num += 1
                ratios[index] += 1
            all_results.append(item)
    corrected_ratio = corrected_num / len(generated)
    print(f'corrected_ratio: {corrected_ratio: .4f}')
        # break
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(all_results, ensure_ascii=False, indent=4))
    print(ratios)

    print(f"Done. Wrote -> {args.output}")

if __name__ == "__main__":
    main()