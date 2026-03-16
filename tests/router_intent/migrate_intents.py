import json

def update_test_cases():
    with open('tests/router_intent/test_cases.json', 'r') as f:
        data = json.load(f)

    for item in data:
        cat = item['category']
        expected_intent = item['expected_analysis']['intent']
        
        if expected_intent == 'clarification':
            # Map category to new intent
            if 'Weak Recognition Language' in cat:
                item['expected_analysis']['intent'] = 'chitchat'
            elif 'Missing Image for Recognition' in cat:
                item['expected_analysis']['intent'] = 'chitchat'
            elif 'Text Depends on Absent Visual Context' in cat:
                item['expected_analysis']['intent'] = 'chitchat'
            elif 'Image Not Food' in cat:
                item['expected_analysis']['intent'] = 'chitchat'
            elif 'Image Too Ambiguous' in cat:
                item['expected_analysis']['intent'] = 'chitchat'
            elif 'Non-Food Image but Asking About Food' in cat:
                item['expected_analysis']['intent'] = 'chitchat'
            elif 'Food Image but Social/Chat Text' in cat:
                item['expected_analysis']['intent'] = 'chitchat'
            elif 'Completely Unrelated Topic' in cat:
                item['expected_analysis']['intent'] = 'chitchat'
            elif 'Minimal/Noisy Input' in cat:
                item['expected_analysis']['intent'] = 'chitchat'
            elif 'Food Safety Question' in cat:
                item['expected_analysis']['intent'] = 'guardrails'
            elif 'Potentially Harmful Food Use' in cat:
                item['expected_analysis']['intent'] = 'guardrails'
            elif 'Self-Harm' in cat:
                item['expected_analysis']['intent'] = 'guardrails'
            elif 'Poisoning or Illegal Use' in cat:
                item['expected_analysis']['intent'] = 'guardrails'
            elif 'Prompt Injection' in cat:
                item['expected_analysis']['intent'] = 'guardrails'
            elif 'Fake System' in cat:
                item['expected_analysis']['intent'] = 'guardrails'
            elif 'Irrelevant event or mission statement' in cat:
                item['expected_analysis']['intent'] = 'chitchat'
            else:
                item['expected_analysis']['intent'] = 'chitchat'
                
    with open('tests/router_intent/test_cases.json', 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    update_test_cases()
