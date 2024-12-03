import re

# CATEGORY_WORDLISTS
global CATEGORY_WORDLISTS 
CATEGORY_WORDLISTS = {
    "Case-Based": [
        "case", "reasoning", "instance", "retrieval", "adaptation", "example", "similarity",
    ],
    "Genetic Algorithms": [
        "genetic", "evolutionary", "population", "mutation", "fitness", "selection", "gene",
    ],
    "Neural Networks": [
        "neural","network", "layer", "gradient", "deep", "weight", "training",
    ],
    "Probabilistic Methods": [
        "probability", "bayesian", "distribution", "inference", "likelihood", "random", "model"
    ],
    "Reinforcement Learning": [
        "reinforcement", "policy", "reward","agent", "action", "state", "environment"
    ],
    "Rule Learning": [
        "rule", "induction", "predicate", "logical", "syntax", "knowledge", "classification",
    ],
    "Theory": [
        "theory", "formal", "proof", "PAC", "complexity", "VC", "hypothesis",
    ],
}

CATEGORY_INDICES = {
    "Case-Based": [827, 489, 1336, 1211, 1022],
    "Genetic Algorithms": [581, 38, 1263, 604, 829],
    "Neural Networks": [299, 140, 495, 368, 310],
    "Probabilistic Methods": [865, 774, 393, 877, 19],
    "Reinforcement Learning": [1254, 647, 51, 821, 474],
    "Rule Learning": [4, 40, 750, 758, 728],
    "Theory": [1057, 485, 1005, 814, 1246]
}

def build_vocabulary(categroy_indices, category_wordlists, vector_length=1433):
    """
    Build a vocabulary from the given category wordlists.

    Args:
        category_wordlists (dict): A dictionary of categories and their word lists.
        vector_length (int): Desired length of the vocabulary.

    Returns:
        tuple: A tuple containing the full vocabulary list and a word-to-index mapping.
    """
    word_to_index = dict()
    for category in category_wordlists.keys():
        for word, index in zip(category_wordlists[category], categroy_indices[category]):
            word_to_index[word] = index
    return word_to_index

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
        if token in word_to_index or (token + 's') in word_to_index:
            feature_vector[word_to_index[token]] = 1
            matched_words.append(token)

    return feature_vector

def process_abstract(abstract, vector_length=1433):
    """
    Process an abstract to generate a binary feature vector.

    Args:
        abstract (str): The input abstract.
        vector_length (int): Length of the feature vector.

    Returns:
        list: A binary feature vector representing the abstract.
    """
    word_to_index = build_vocabulary(CATEGORY_INDICES, CATEGORY_WORDLISTS, vector_length=vector_length)
    feature_vector = generate_binary_feature_vector(abstract, vector_length=vector_length, word_to_index=word_to_index)
    return feature_vector
