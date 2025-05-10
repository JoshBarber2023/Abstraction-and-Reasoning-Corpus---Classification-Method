import math

# This function calculates a Solomonoff score, which is commonly used in algorithmic complexity and 
# information theory. The score is used to measure the complexity of a sequence or system based on prior
# probabilities and complexity. The Solomonoff score is inherently related to the concept of compression:
# the more likely a sequence is, the lower its complexity, and the higher its score. 
#
# A logarithmic scale is used because it allows us to compress the difference between highly likely and
# unlikely events in a manageable way. The log function helps scale the scores to a more manageable range,
# especially when dealing with very small or very large probabilities, and ensures that penalties for 
# unlikely events grow slowly, which is critical in models that involve uncertainty or probabilistic reasoning.

# BONUSES FROM LOG:
# - A linear penalty for multiple failures.
# A bounded increase from rule successes.
# Prevents a single rule from dominating.


def calculate_solomonoff_score(passed_list, prior, complexity):
    log_score = 0.0
    for passed in passed_list:
        if passed:
            log_score += math.log(prior / complexity + 1e-8)  # avoid log(0)
        else:
            log_score += math.log(1)  # consistent penalty
    return -log_score


