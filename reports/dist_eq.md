# NLL for distributions ready to enter into Wolfram Alpha

## Lognormal with location paramter (mu, sigma, loc) -> (m, exp(s), l)
d/dm -ln(1/((x-l) * exp(s)) * exp(-1/2 * (ln(x-l) - m)^2 / exp(2s)))
d/ds -ln(1/((x-l) * exp(s)) * exp(-1/2 * (ln(x-l) - m)^2 / exp(2s)))
d/dl -ln(1/((x-l) * exp(s)) * exp(-1/2 * (ln(x-l) - m)^2 / exp(2s)))

d^2/dmdm -ln(1/((x-l) * exp(s)) * exp(-1/2 * (ln(x-l) - m)^2 / exp(2s)))
d^2/dsds -ln(1/((x-l) * exp(s)) * exp(-1/2 * (ln(x-l) - m)^2 / exp(2s)))
d^2/dldl -ln(1/((x-l) * exp(s)) * exp(-1/2 * (ln(x-l) - m)^2 / exp(2s)))
d^2/dsdm -ln(1/((x-l) * exp(s)) * exp(-1/2 * (ln(x-l) - m)^2 / exp(2s)))
d^2/dldm -ln(1/((x-l) * exp(s)) * exp(-1/2 * (ln(x-l) - m)^2 / exp(2s)))
d^2/dlds -ln(1/((x-l) * exp(s)) * exp(-1/2 * (ln(x-l) - m)^2 / exp(2s)))

## SHASH (four parameter distribution)
l = loc
s = ln(scale)
n = nu
t = ln(tau)

-loglik: 
z = (x - loc)/scale
-ln(cosh(exp(t)*asinh(z) - n)*exp(t-s) + 0.5*(ln(2*pi*(1+z^2)) + sinh(exp(t)*asinh(z) - n))

s-t -ln(cosh(exp(t)*asinh((x-l)/exp(s)) - n)) + 0.5*(ln(2*pi*(1+(x-l)^2/exp(2s))) + sinh(exp(t)*asinh((x-l)/exp(s)) - n)^2)

Gradients:
z = (x - loc)/scale
r = nu - tau*asinh(z)
w = sinh(r)*cosh(r) - tanh(r)
v = 1/(z*z + 1)
u = tau*w*sqrt(v)

d/d(loc):       u/scale - z*z*v
d/d(s):         z*u + v
d/d(nu):        w
d/d(t):         -tau*asinh(z)*w - 1

## Gamma with location parameter
