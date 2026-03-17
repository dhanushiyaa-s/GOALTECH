def stable(history, key):
    vals = [h[key] for h in history]
    return sum(vals) > len(vals)*0.6 if vals else False