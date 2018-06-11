# -*- coding: utf-8 -*-


def _get_character_pairs(text):
    results = dict()
    for word in text.split():
        for pair in [word[i] + word[i + 1] for i in range(len(word) - 1)]:
            if pair in results:
                results[pair] += 1
            else:
                results[pair] = 1
    return results


def strike_a_match(string1, string2):
    s1_pairs = _get_character_pairs(string1)
    s2_pairs = _get_character_pairs(string2)
    s1_size = sum(s1_pairs.values())
    s2_size = sum(s2_pairs.values())
    if s1_size == 0 and s2_size == 0: return 0.
    intersection_count = 0
    # determine the smallest dict to optimise the calculation of the
    # intersection.
    if s1_size < s2_size:
        smaller_dict = s1_pairs
        larger_dict = s2_pairs
    else:
        smaller_dict = s2_pairs
        larger_dict = s1_pairs
    # determine the intersection by counting the subtractions we make from both
    # dicts.
    for pair, smaller_pair_count in list(smaller_dict.items()):
        if pair in larger_dict and larger_dict[pair] > 0:
            if smaller_pair_count < larger_dict[pair]:
                intersection_count += smaller_pair_count
            else:
                intersection_count += larger_dict[pair]
    res = (2.0 * intersection_count) / (s1_size + s2_size)
    return res
