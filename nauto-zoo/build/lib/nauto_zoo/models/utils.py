def infer_confidence(score: float, threshold: float, low_bounary: float = 0., high_bounday: float = 1.) -> int:
    if score <= threshold:
        relative_score = (threshold - score) / (threshold - low_bounary)
    else:
        relative_score = (score - threshold) / (high_bounday - threshold)
    return int(100 * relative_score)
