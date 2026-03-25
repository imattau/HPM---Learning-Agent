def make_uniform(D=3, K=4):
    probs = np.ones((D, K)) / K
    return CategoricalPattern(probs, K=K)