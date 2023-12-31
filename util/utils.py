import re
import json
import csv
import matplotlib.pyplot as plt

def get_positions(numbers, target):
    positions = []
    start = None
    for i, num in enumerate(numbers):
        if num == target:
            if start is None:
                start = i
        elif start is not None:
            positions.append((start, i - 1))
            start = None
    if start is not None:
        positions.append((start, len(numbers) - 1))
    return positions
def get_different_num_positions(nums):
    positions = {}
    start = 0
    for i in range(1, len(nums)):
        if nums[i] != nums[i-1]:
            if nums[i-1] not in positions:
                positions[nums[i-1]] = []
            positions[nums[i-1]].append((start, i-1))
            start = i
    if nums[-1] not in positions:
        positions[nums[-1]] = []
    positions[nums[-1]].append((start, len(nums)-1))
    return positions
def replace_with_neg1(nums, mask):
    for i in range(len(nums)):
        for j in range(len(nums[i])):
            if mask[i][j] == 0:
                nums[i][j] = -1
    return nums
def print_2dlist_to_file(my_list, filename):
    with open(filename, 'w') as f:
        for sublist in my_list:
            for item in sublist:
                f.write("%s " % item)
            f.write("\n")
def read_2dintlist_from_file(filename):
    # Open the file for reading
    with open(filename, 'r') as f:
        # Initialize the two-dimensional list
        numbers = []
        # Loop over each line in the file
        for line in f:
            # Split the line into individual numbers
            row = line.strip().split()
            # Convert the numbers to integers
            row = [int(num) for num in row]
            # Add the row to the two-dimensional list
            numbers.append(row)
    return numbers
def read_2dstrlist_from_file(filename):
    # Open the file for reading
    with open(filename, 'r',encoding='utf-8') as f:
        # Initialize the two-dimensional list
        numbers = []
        # Loop over each line in the file
        for line in f:
            # Split the line into individual numbers
            row = line.strip().split()
            # Add the row to the two-dimensional list
            numbers.append(row)
    return numbers
def get_token_labels(token_pos_dict, tokens):
    labels = {}
    for key in token_pos_dict:
        if key != -1:
            label_tokens = []
            for pos in token_pos_dict[key]:
                label_tokens.append('|'.join(tokens[pos[0]:pos[1]+1]))
            #combined_tokens = '|'.join(label_tokens)
            labels[key] = label_tokens
    return labels
def getPRTokenLabels(token_pos_list,tokens):
    labels4sentence = []
    for pos in token_pos_list:
        t = tokens[pos[0]:pos[1]+1]
        labels4sentence.append(('|'.join(t),pos))
    return labels4sentence
def getLabels4TokensList(token_pos_dict_list, tokens_list):
    labels_list = []
    for i in range(len(tokens_list)):
        t = get_token_labels(token_pos_dict_list[i],tokens_list[i])
        labels_list.append(t)
    return labels_list
def getPRPosFromPattern(lst):
    pattern = '01*'
    matches = [(m.start(), m.end()-1) for m in re.finditer(pattern, ''.join(map(str, lst)))]
    return matches
def convert_negatives(lst):
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            if lst[i][j] == -1:
                lst[i][j] = 3
    return lst
def extractPredicate(filename):
    p = set()
    with open(filename, encoding='utf-8') as f:
        data = json.load(f)
    for sen in data:
        for proposition in sen['labels']:
            p.add(proposition['REL'])
    return p 
def filterPredicate(predicate_list,pset):
    """
    This function does something with two parameters.

    :param param1: a 2 dim predicte list, where 2 dim hold several predicates in the form of (predicte,pos) (e.g. ("发|送",[4,5]))
    :type param1: list 
    :param param2: a set of predicate text
    :type param2: str
    :return: The 2 dim filtered predicate list, in the same form of "param 1"
    :rtype: float
    """
    filtered_list = []
    for predicates in predicate_list:
        t = []
        # for p in [s[0].replace('|', '') for s in predicates]:
        #     if p in pset:
        #         t.append(s)
        for p in predicates:
            ptext = p[0].replace('|','')
            if ptext in pset:
                t.append(p)        
        filtered_list.append(t)
    return filtered_list
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

def write_loss_values_to_csv(loss_values, output_file):
    # Open the CSV file in write mode
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(["Loss"])

        # Write each loss value as a row in the CSV file
        for loss in loss_values:
            writer.writerow([loss])
def append_loss_values_to_csv(loss_values, output_file):
    # Open the CSV file in append mode
    with open(output_file, "a", newline="") as file:
        writer = csv.writer(file)

        # Write each loss value as a row in the CSV file
        for loss in loss_values:
            writer.writerow([loss])

def draw_and_save_loss_curve(loss_list, file_path):
    # create a figure and axis object
    fig, ax = plt.subplots()
    # plot the loss curve
    ax.plot(loss_list)
    # set the axis labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve')

    # save the plot to a file
    fig.savefig(file_path)

    # show the plot
    plt.show()
def read_list_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data_list = [row[0] for row in reader]
    return data_list