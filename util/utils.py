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
def print_2dlist_to_file(my_list, filename):
    with open(filename, 'w') as f:
        for sublist in my_list:
            for item in sublist:
                f.write("%s " % item)
            f.write("\n")