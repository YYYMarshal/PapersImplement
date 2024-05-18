
def Normalization(X, Min, Max, a, b):
    X = (X - Min) / ((Max - Min) + (1e-8))
    X = X * (b - a) + a
    return X

