def classify(attention_score, phone_detected):

    if phone_detected:
        return "Using Phone"

    elif attention_score is None:
        return "Unknown"

    elif attention_score < 0.5:
        return "Not Attentive"

    else:
        return "Attentive"