def compute_score(s_phone, s_eye, s_talk, s_gaze):
    score = 100
    if s_phone: score -= 60
    if s_eye: score -= 50
    if s_talk: score -= 25
    if not s_gaze: score -= 15
    return max(0, min(100, score))


def classify_behavior(s_eye, s_phone, s_talk, silent, s_gaze):
    if s_eye:
        return "SLEEPING"
    elif s_phone:
        return "USING PHONE"
    elif s_talk:
        return "TALKING"
    elif silent:
        return "QUIET"
    elif not s_gaze:
        return "NOT FOCUSED"
    else:
        return "ATTENTIVE"