import json

def extract_arguments(json_file):
    with open(json_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    # p_arguments = [['ARG':,'ARGM']]
    p_arguments = []
    
    for sen_info in data:
        labels = sen_info.get('labels', [])
        for proposition in labels:
            arguments = {}
            args = proposition.get('ARG', {})
            argms = proposition.get('ARGM', {})
            arguments.update(args)
            arguments.update(argms)
            p_arguments.append(arguments)
    return p_arguments
json_file = './data/data_correct_formated.json'
arguments = extract_arguments(json_file)

# Print the extracted ARG and ARGM values
for arg in arguments:
    print(arg)