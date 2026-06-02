import math
def get_lr_cosine(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        alpha_t = t / T_w * alpha_max
    elif t<= T_c and t>= T_w:
        alpha_t = alpha_min + (1 + math.cos(math.pi / (T_c - T_w) * (t - T_w) )) / 2 * (alpha_max - alpha_min)
    elif t > T_c:
        alpha_t = alpha_min
    else :
        raise ValueError(f"t in out of range ({T_w}, {T_c})")
    return alpha_t
