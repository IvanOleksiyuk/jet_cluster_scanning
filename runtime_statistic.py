import numpy as np
def filter_lines_with_word_generator(filename, word):
    with open(filename, 'r') as file:
        for line in file:
            if word in line:
                yield line.strip()

if __name__ == "__main__":
    test_file = "jobs/outputs/output_job_4396247.txt"  # Place your logfile here 
    word_to_find = "train"	

    list = [float(i[18:][:-16]) for i in filter_lines_with_word_generator(test_file, word_to_find)]
    #print(list)
    print("Mean: ", np.mean(list[:100]))
    print(len(list))
    