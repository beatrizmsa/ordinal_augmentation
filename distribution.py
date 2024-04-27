import sympy
n, media, var, alpha, beta = sympy.symbols('n media var alpha beta')
d = sympy.solve([(n*alpha)/(alpha+beta)-media, (((n*alpha*beta)*(alpha+beta+n))/( ((alpha+beta)**2) * (alpha+beta+1))) - var], [alpha, beta], dict=True)
print(d[0][alpha].subs({media: 5, n: 10, var: 0.00001}))
# -5.00001800007200
