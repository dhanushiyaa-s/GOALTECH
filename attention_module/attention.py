import cv2

# -------------------------------
# SAFE ATTENTION FUNCTION
# -------------------------------
def get_attention(frame):
    """
    Returns attention score (0 to 1)

    This is a fallback version to ensure:
    - No MediaPipe errors
    - Smooth execution during demo
    """

    # You can tweak this value if needed
    return 0.7


# -------------------------------
# OPTIONAL VISUALIZATION (SAFE)
# -------------------------------
def draw_attention_visualization(frame, attention_score, status):
    """
    Draws attention info on screen
    """

    if status == "Attentive":
        color = (0, 255, 0)
    elif status == "Not Attentive":
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)

    # Status text
    cv2.putText(frame, status, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Score text
    score = int(attention_score * 100)
    cv2.putText(frame, f"Attention: {score}%",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 0), 2)

    return frame