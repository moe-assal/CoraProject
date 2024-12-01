# CATEGORY_WORDLISTS
CATEGORY_WORDLISTS = {
    "Case-Based": [
        "case", "reasoning", "instance", "retrieval", "adaptation", "example", "similarity",
    ],
    "Genetic Algorithms": [
        "genetic", "evolutionary", "chromosome", "mutation", "fitness", "selection", "gene",
    ],
    "Neural Networks": [
        "neural", "networks","network", "layers","layer","activation", "backpropagation", "perceptron", "gradient",
        "deep", "weights", "bias", "input", "output", "hidden", "connection", "training",
    ],
    "Probabilistic Methods": [
        "probability", "bayesian", "distribution", "inference", "likelihood", "random", "model",
    ],
    "Reinforcement Learning": [
        "reinforcement", "policy", "reward","agent", "agents", "action", "state", "environment", "rewards",
    ],
    "Rule Learning": [
        "rule", "induction", "predicate", "logical", "syntax", "knowledge", "classification",
    ],
    "Theory": [
        "theory", "formal", "proof", "axiom", "foundation", "principle", "hypothesis",
    ],
}



# Build vocabulary and mapping
def build_vocabulary(category_wordlists, vector_length=1433):
    full_vocabulary = list({word for words in category_wordlists.values() for word in words})
    if len(full_vocabulary) < vector_length:
        full_vocabulary.extend([f"dummy_word_{i}" for i in range(vector_length - len(full_vocabulary))])
    elif len(full_vocabulary) > vector_length:
        full_vocabulary = full_vocabulary[:vector_length]
    word_to_index = {word: idx for idx, word in enumerate(full_vocabulary)}
    return full_vocabulary, word_to_index

def generate_binary_feature_vector(abstract, vector_length, word_to_index):
    feature_vector = [0] * vector_length
    tokens = abstract.lower().split()

    print("Tokens in Abstract:", tokens)  # Debugging: Tokenized words

    matched_words = []
    for token in tokens:
        if token in word_to_index:
            feature_vector[word_to_index[token]] = 1
            matched_words.append(token)

    print("Matched Words:", matched_words)  # Debugging: Words that matched the vocabulary
    return feature_vector

def process_abstract(abstract):
    full_vocabulary, word_to_index = build_vocabulary(CATEGORY_WORDLISTS, vector_length=1433)
    feature_vector = generate_binary_feature_vector(abstract, vector_length=1433, word_to_index=word_to_index)
    return feature_vector
