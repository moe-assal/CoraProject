import re

# CATEGORY_WORDLISTS
global CATEGORY_WORDLISTS 
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



def build_vocabulary(category_wordlists, vector_length=1433):
    """
    Build a vocabulary from the given category wordlists.

    Args:
        category_wordlists (dict): A dictionary of categories and their word lists.
        vector_length (int): Desired length of the vocabulary.

    Returns:
        tuple: A tuple containing the full vocabulary list and a word-to-index mapping.
    """
    if not category_wordlists:
        raise ValueError("CATEGORY_WORDLISTS cannot be empty.")
    
    full_vocabulary = list({word for words in category_wordlists.values() for word in words})
    if len(full_vocabulary) < vector_length:
        full_vocabulary.extend([f"__dummy_word_{i}__" for i in range(vector_length - len(full_vocabulary))])
    elif len(full_vocabulary) > vector_length:
        full_vocabulary = full_vocabulary[:vector_length]
    
    word_to_index = {word: idx for idx, word in enumerate(full_vocabulary)}
    return full_vocabulary, word_to_index

def generate_binary_feature_vector(abstract, vector_length, word_to_index):
    """
    Generate a binary feature vector based on the presence of words in the abstract.

    Args:
        abstract (str): The input abstract.
        vector_length (int): Length of the feature vector.
        word_to_index (dict): A mapping from words to indices in the vector.

    Returns:
        list: A binary feature vector.
    """
    feature_vector = [0] * vector_length
    tokens = re.findall(r'\b\w+\b', abstract.lower())  # Robust tokenization

    matched_words = []
    for token in tokens:
        if token in word_to_index:
            feature_vector[word_to_index[token]] = 1
            matched_words.append(token)

    return feature_vector

def process_abstract(abstract, vector_length=1433):
    print("abstract",abstract)
    """
    Process an abstract to generate a binary feature vector.

    Args:
        abstract (str): The input abstract.
        category_wordlists (dict): A dictionary of categories and their word lists.
        vector_length (int): Length of the feature vector.

    Returns:
        list: A binary feature vector representing the abstract.
    """
    full_vocabulary, word_to_index = build_vocabulary(CATEGORY_WORDLISTS, vector_length=vector_length)
    feature_vector = generate_binary_feature_vector(abstract, vector_length=vector_length, word_to_index=word_to_index)
    return feature_vector