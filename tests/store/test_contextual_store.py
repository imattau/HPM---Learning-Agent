def make_grid(rows, cols, colors):
    """Helper: grid with specified dimensions and colour values."""
    g = np.zeros((rows, cols), dtype=int)
    for (i, c) in enumerate(colors):
        g[i % rows, i % cols] = c
    return g