import json

with open('../../../self_explain_examples/1.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for index, line in enumerate(lines):
    json_data = json.loads(line)
    with open(f'tmp/{index}.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)