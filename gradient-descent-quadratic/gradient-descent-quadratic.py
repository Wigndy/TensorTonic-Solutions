def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    # Write code here
    ans = x0
    for i in range(steps):
        grad =   2* a*ans + b
        ans = ans - grad * lr
    return ans
