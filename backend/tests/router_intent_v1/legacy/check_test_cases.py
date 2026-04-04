# 去掉test_cases.json中的全部的description键，保存到源文件

import json

def remove_description(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    remove_description_recursive(data)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def remove_description_recursive(data):
    if isinstance(data, dict):
        if 'description' in data:
            del data['description']
        for key in data:
            remove_description_recursive(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_description_recursive(item)


def watch_text(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    for line in data:
        print("text:", line['input']['text'])
        print("intent:", line['expected_analysis']['intent'])

if __name__ == "__main__":
    file_path = './tests/router_intent/test_cases.json'
    watch_text(file_path)



