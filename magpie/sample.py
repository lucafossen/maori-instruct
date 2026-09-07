import random
import json


with open('instruct_translation/output/complete/Magpie-Llama-3.1-8B-Instruct-Filtered_translated.jsonl', 'r') as rf:
    data = [json.loads(line) for line in rf.readlines()]
    random.shuffle(data)
    with open('instruct_translation/output/complete/Magpie-Llama-3.1-8B-Instruct-Filtered_translated_shuffled_1M.jsonl', 'w') as wf:
        for i in range(1_000_000):
            print(json.dumps(data[i], ensure_ascii=False), file=wf)