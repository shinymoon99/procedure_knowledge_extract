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
                label_tokens.extend(tokens[pos[0]:pos[1]+1])
            combined_tokens = '|'.join(label_tokens)
            if key in labels:
                labels[key].append(combined_tokens)
            else:
                labels[key] = [combined_tokens]
    return labels
def getLabels4TokensList(token_pos_dict_list, tokens_list):
    labels_list = []
    for i in range(len(tokens_list)):
        t = get_token_labels(token_pos_dict_list[i],tokens_list[i])
        labels_list.append(t)
    return labels_list