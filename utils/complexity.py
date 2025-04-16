def rule_complexity(rule_func):
    return len(rule_func.__code__.co_code)
