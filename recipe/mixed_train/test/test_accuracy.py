import os
import json
import re
import time
import random
import requests
import concurrent.futures as cf
from collections import Counter
from typing import Optional, List
# 定义模型和API配置
model = "Kimi-K2-Instruct"  # 使用的模型，其他模型名请直接复制https://api.probex.top/pricing页面对应的模型名
api_key = "sk-A8HW3cVbVxP7x3idg6BEq02WgyKiQX3N8DekE8ymKZo8EEAc"  # 替换为你自己的API Key

# 定义请求头
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def send_message(message):
    # 定义请求数据（JSON格式）
    data = {
        "model": model,  # 使用前面定义的模型
        "messages": [
            {
                "role": "user",  # 用户发送的消息
                "content": message  # 消息内容
            }
        ],
        "stream": False  # 是否启用流式响应（False 表示一次性返回完整数据，True 表示流式返回数据）
    }

    # 发送POST请求
    try:
        response = requests.post(
            "https://api.probex.top/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=300
        )

        # 处理非流式响应
        if response.status_code == 200:
            result = response.json()  # 解析JSON响应
            content = result['choices'][0]['message']['content']  # 提取content字段
            print(content)  # 直接打印content内容
            return content
        else:
            print("请求失败，状态码:", response.status_code)
            print("错误信息:", response.text)

    except requests.exceptions.Timeout:
        print("请求超时：服务器没有响应")
    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {e}")

def judge_answer(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        prefix_answer = data['prefix_answer'].split('Your task is to understand a given standard problem solving process'
                                                    ' of a given question, then finish an incomplete reasoning process. '
                                                    'The question is :\n')[1]
        question = prefix_answer.split('The standard solving process is as followings:')[0].strip()
        prefix_answer = prefix_answer.split('The standard solving process is as followings:')[1]
        standard_process = prefix_answer.split('User: **Finish the following incomplete answer**:')[0].strip()
        incomplete_answer = prefix_answer.split('User: **Finish the following incomplete answer**:')[1].strip()
        # print(incomplete_answer)
        answer_before = data['answer_before']
        answer_after = data['answer_after']
        step_before = answer_before.split(incomplete_answer)[1]
        # print(step_before)
        step_after = answer_after.split(incomplete_answer)[1]
        # print(step_after)
        message = (f"我需要你认真地帮我解决以下问题。我现在有一个数学推理问题：\n{question} \n这个问题的答案是：{standard_process}\n"
                   f"我现在让一个功能相对弱的模型回答了这个问题，模型先回答了一部分如下：\n{incomplete_answer} \n 现在有两个可能"
                   f"紧接着以上不完整的解答的步骤，请你判断以下两个步骤是否能导致模型回答的答案正确，请用数字1表示只有第一个步骤正确，"
                   f"数字2表示只有第二个步骤正确，数字3表示两个步骤都正确，数字4表示两个步骤都不正确：\n第一个步骤：\"{step_before}\" \n"
                   f"第二个步骤：\"{step_after}\" ")
        print(message)
        result = send_message(message)
        sentences_of_result = result.split('\n')
        sentences_of_result.reverse()
        label = 0
        for sentence in sentences_of_result:
            if '1' in sentence:
                label = 1
                break
            elif '2' in sentence:
                label = 2
                break
            elif '3' in sentence:
                label = 3
                break
            elif '4' in sentence:
                label = 4
                break
            else:
                continue
        print('label: ', label)
        return label

path_prefix = 'parsed_coef_0.1/1/'
paths = os.listdir(path_prefix)
all_results = []
all_types_results = {1: 0, 2: 0, 3: 0, 4: 0}
for path in paths:
    start_time = time.time()
    answer_label = judge_answer(path_prefix + path)
    end_time = time.time()
    print(f"用时：{end_time - start_time}")
    all_results.append(answer_label)
    all_types_results[answer_label] += 1
    # exit(0)
    if len(all_results) > 10:
        break