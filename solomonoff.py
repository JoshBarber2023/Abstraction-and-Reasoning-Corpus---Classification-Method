import math

def calculate_solomonoff_score(passed_list, prior, complexity):
    """
    Improved Solomonoff score calculation with better handling
    
    Args:
        passed_list: List of boolean test results
        prior: Prior probability of the hypothesis
        complexity: Complexity measure of the hypothesis
    
    Returns:
        Negative log score (lower is better)
    """
    if not passed_list:
        return float('inf')  # No tests means infinite bad score
    
    # Ensure reasonable bounds
    prior = max(min(prior, 0.99), 0.01)  # Clamp between 0.01 and 0.99
    complexity = max(complexity, 1)      # Minimum complexity of 1
    
    log_score = 0.0
    
    for passed in passed_list:
        if passed:
            # Reward for passing: log(prior/complexity) 
            # Higher prior and lower complexity give better scores
            success_prob = prior / complexity
            log_score += math.log(success_prob + 1e-8)  # Add small epsilon to avoid log(0)
        else:
            # Penalty for failing: log(1 - prior/complexity)
            # This creates a penalty that scales with how confident we were
            success_prob = prior / complexity
            failure_prob = 1.0 - min(success_prob, 0.99)  # Ensure failure_prob > 0
            log_score += math.log(max(failure_prob, 0.01))  # Ensure positive argument
    
    # Return negative log score (so lower scores are better)
    return -log_score

def normalize_solomonoff_scores(scores):
    """
    Normalize a list of Solomonoff scores to probabilities
    
    Args:
        scores: List of Solomonoff scores (lower is better)
    
    Returns:
        List of normalized probabilities (higher is better)
    """
    if not scores:
        return []
    
    # Convert to positive values (invert since lower scores are better)
    max_score = max(scores)
    inverted_scores = [max_score - score + 1e-8 for score in scores]
    
    # Softmax normalization
    exp_scores = [math.exp(score - max(inverted_scores)) for score in inverted_scores]
    total = sum(exp_scores)
    
    return [score / total for score in exp_scores]