"""
Examples showing usage of the Scolx Math API with both plain text and LaTeX expressions.
"""

from scolx_math.advanced_latex import (
    differentiate_latex_with_steps,
    integrate_latex_with_steps,
    limit_latex_with_steps,
    series_latex_with_steps,
    solve_equation_latex_with_steps,
)


def example_plain_text():
    """Examples using plain text mathematical expressions."""
    print("=== Plain Text Examples ===")

    # Integration example
    from scolx_math.explain.explainers import integrate_with_steps

    result, steps = integrate_with_steps("x**2", "x")
    print(f"Integral of x^2: {result}")
    print(f"Steps: {steps}")
    print()

    # Differentiation example
    from scolx_math.core.operations import differentiate_expr

    result = differentiate_expr("x**3", "x")
    print(f"Derivative of x^3: {result}")
    print()

    # Equation solving example
    from scolx_math.core.operations import solve_equation

    result = solve_equation("x**2 - 4", "x")
    print(f"Solutions to x^2 - 4 = 0: {result}")
    print()

    # Simplification example
    from scolx_math.core.operations import simplify_expr

    result = simplify_expr("(x+1)**2 - x**2 - 2*x - 1")
    print(f"Simplification of (x+1)^2 - x^2 - 2*x - 1: {result}")
    print()


def example_latex():
    """Examples using LaTeX mathematical expressions."""
    print("=== LaTeX Examples ===")

    # Integration with LaTeX
    result, steps = integrate_latex_with_steps("x^2", "x")
    print(f"Integral of x^2 (from LaTeX): {result}")
    print(f"Steps: {steps}")
    print()

    # Differentiation with LaTeX
    result, steps = differentiate_latex_with_steps("x^3", "x")
    print(f"Derivative of x^3 (from LaTeX): {result}")
    print(f"Steps: {steps}")
    print()

    # Equation solving with LaTeX
    result, steps = solve_equation_latex_with_steps("x^2 - 4", "x")
    print(f"Solutions to x^2 - 4 = 0 (from LaTeX): {result}")
    print(f"Steps: {steps}")
    print()

    # Limit with LaTeX
    result, steps = limit_latex_with_steps("\\frac{\\sin(x)}{x}", "x", "0")
    print(f"Limit of sin(x)/x as x approaches 0 (from LaTeX): {result}")
    print(f"Steps: {steps}")
    print()

    # Series expansion with LaTeX
    result, steps = series_latex_with_steps("e^x", "x", "0", 5)
    print(f"Series expansion of e^x around 0 (from LaTeX): {result}")
    print(f"Steps: {steps}")
    print()


if __name__ == "__main__":
    example_plain_text()
    example_latex()
    print("All examples completed successfully!")
