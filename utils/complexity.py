import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

def rule_complexity(rule_description: str) -> int:
    tokens = tokenizer.encode(rule_description)
    return len(tokens)
