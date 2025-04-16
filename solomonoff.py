def calculate_solomonoff_score(passed_list, prior, complexity):
    score = 1.0
    for passed in passed_list:
        if passed:
            score *= prior / complexity
        else:
            score *= 0.01  # Penalize for failing
    return score
