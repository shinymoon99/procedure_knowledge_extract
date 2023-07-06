def print_2dlist_to_file(my_list, filename):
    with open(filename, 'w') as f:
        for sublist in my_list:
            for item in sublist:
                f.write("%s " % item)
            f.write("\n")