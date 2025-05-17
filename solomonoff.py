import math
def calculate_solomonoff_score(passed_list, prior, complexity):
    log_score = 0.0
    for passed in passed_list:
        if passed:
            log_score += math.log(prior / complexity + 1e-8)  # avoid log(0)
        else:
            log_score += math.log(1)  # consistent penalty
    return -log_score


