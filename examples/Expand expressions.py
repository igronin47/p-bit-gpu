import sympy as sp

N0, N1 ,N2, N3, A0, A1, B0, B1 = sp.symbols('N0 N1 N2 N3 A0 A1 B0 B1')
expr = N3 + N2 +N1 + N0 - (A1 + A0)*(B1 + B0)
expanded_expr = sp.expand(expr**2)
print(expanded_expr)
