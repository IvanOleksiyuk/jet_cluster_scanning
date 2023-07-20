import math
import numpy as np
import scipy.stats as stats

# statistical functions
def p2Z(p):
    if isinstance(p, np.ndarray):
        out = []
        for one_p in p:
            out.append(-stats.norm.ppf(one_p))
        return np.array(out)
    elif isinstance(p, list):
        out = []
        for one_p in p:
            out.append(-stats.norm.ppf(one_p))
        return np.array(out)
    else:
        return -stats.norm.ppf(p)

def p_value(stat, tstat_list):
    return (
        (np.sum(tstat_list >= stat) + np.sum(tstat_list > stat)) / 2 / len(tstat_list)
    )

# operatios with dictionaries

def make_lists_in_dict(dictionary):
    for key in dictionary:
        dictionary[key] = [dictionary[key]]
    return dictionary


def append_dict_to_dict(dictionary, dictionary_to_append):
    for key in dictionary:
        dictionary[key].append(dictionary_to_append[key])
    return dictionary


def add_lists_in_dicts(dict1, dict2):
    if dict1 == {}:
        return dict2
    elif dict2 == {}:
        return dict1
    else:
        for key in dict1:
            dict1[key] = dict1[key] + dict2[key]
        return dict1

# test statistic ensambling operations

def ensamble_means(arr, k):
    return np.mean(arr.reshape(-1, k), axis=1)


def ensamble_stds(arr, k):
    return np.std(arr.reshape(-1, k), axis=1)


def ensamble_medians(arr, k):
    return np.median(arr.reshape(-1, k), axis=1)


############################ UNUSED ###########################################


def divide_into_batches(array, batch_size, drop_last=True):
    num_batches = math.ceil(len(array) / batch_size)
    batches = []
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch = array[start:end]
        if drop_last and len(batch) < batch_size:
            break
        batches.append(batch)
    return batches


def calculate_batch_averages(batches):
    averages = []
    for batch in batches:
        average = sum(batch) / len(batch)
        averages.append(average)
    return averages
