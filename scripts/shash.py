"""The NGBoost SHASH distribution and scores"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.special import kn

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore


class SHASHLogScore(LogScore):
    """
    MLE score with metric (Fisher Information Matrix) calculated using
    random sampling defined in the parent class.
    """

    def score(self, Y):
        return -self.logpdf(Y)

    def d_score(self, Y):
        x = [self.loc, np.log(self.scale), self.nu, np.log(self.tau)]
        return self.grad_nll(x, Y)

    def score2(self, Y):
        return -self.logpdf(Y)

    def d_score2(self, Y):
        z = (Y - self.loc) / self.scale
        r = self.nu - self.tau * np.arcsinh(z)
        w = np.sinh(r) * np.cosh(r) - np.tanh(r)
        v = 1./(z*z + 1.)
        u = self.tau * w * np.sqrt(v)

        D = np.zeros((self.scale.shape[0], 4))
        D[:, 0] = u/self.scale - z*z*v     
        D[:, 1] = z*u + v
        D[:, 2] = w
        D[:, 3] = -self.tau * np.arcsinh(z) * w - 1.
        return D


class SHASH(RegressionDistn):
    """
    Implements the Sinh-ArcSinh (SHASH) distribution for NGBoost.

    The SHASH distribution defined in Jones & Pewsey (2009, p2)
    has four parameters: loc, scale, nu, and tau,
    defining the location, scale, skewness, and tail width.

    This distribution has LogScore implemented using the default metric method.
    """

    n_params = 4
    scores = [SHASHLogScore]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self._params = params
        self.loc = params[0]
        self.scale = np.exp(params[1])
        self.nu = params[2]
        self.tau = np.exp(params[3])

    def __getattr__(self, name):
        if name in dir(self):
            return getattr(self, name)
        return None

    def mean(self):
        q = 1/self.tau
        # Modified Bessel functions of the second kind
        Kp = kn(0.25*(q + 1), 0.25)  
        Km = kn(0.25*(q - 1), 0.25)
        P = np.exp(0.25)/np.sqrt(8*np.pi)*(Kp + Km)
        return self.loc + self.scale * np.sinh(self.nu*q) * P
    
    def sample(self, m):
        """Sample SHASH pdf: follows R gamlss rSHASHo"""
        z = np.random.randn(m)  # Sample from standard normal
        return self.loc + self.scale*np.sinh((np.arcsinh(z) + self.nu)/self.tau)

    @staticmethod
    def loglik(Y, loc, scale, nu, tau):
        """Negative log likelihood for shash: follows R gamlss dSHASHo"""
        z = (Y - loc)/scale
        w = tau * np.arcsinh(z) - nu
        s = np.sinh(w)
        c = np.sqrt(1 + s*s)
        return np.log(c*tau/scale) - 0.5*(np.log(2*np.pi*(1 + z*z)) + s*s)

    @staticmethod
    def nll(x, Y):
        """Negative loglikelihood with the interface for scipy minimize"""
        return -self.loglik(Y, x[0], np.exp(x[1]), x[2], np.exp(x[3]))

    @staticmethod
    def nllx(x, Y):
        """Negative loglikelihood with the interface for scipy minimize"""
        loc, scale, nu, tau = x
        scale = np.exp(scale)
        tau = np.exp(tau)

        z = (Y - loc)/scale
        w = tau * np.arcsinh(z) - nu
        s = np.sinh(w)
        c = np.sqrt(1 + s*s)
        return -(np.log(c*tau/scale) - 0.5*(np.log(2*np.pi*(1 + z*z)) + s*s))

    @staticmethod
    def grad_nll(x, Y):
        """Gradient of nll with the interface for scipy minimize"""
        loc, scale, nu, tau = x
        scale = np.exp(scale)
        tau = np.exp(tau)

        z = (Y - loc) / scale
        r = nu - tau * np.arcsinh(z)
        w = np.sinh(r) * np.cosh(r) - np.tanh(r)
        v = 1./(z*z + 1.)
        u = tau * w * sqrt(v)

        D = np.zeros((scale.shape[0], 4))
        D[:, 0] = u/scale - z*z*v     
        D[:, 1] = z*u + v
        D[:, 2] = w
        D[:, 3] = -tau * np.arcsinh(z) * w - 1.
        return D

    def logpdf(self, Y):
        return self.loglik(Y, self.loc, self.scale, self.nu, self.tau)

    def logpdfx(self, Y):
        x = [self.loc, np.log(self.scale), self.nu, np.log(self.tau)]
        return -self.nllx(x, Y)

    def fit(Y):
        """Minimize negative loglikelihood of the observations Y"""
        pars0 = [self.loc, np.log(self.scale), self.nu, np.log(self.tau)]
        res = minimize(nll, pars0, args = (Y,), method = 'BFGS', jac = grad_nll)
        return res.x

    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale,
                "nu": self.nu, "tau": self.tau)}


    def fit(Y):
        """Minimize negative loglikelihood of the observations Y"""
        pars0 = [self.loc, np.log(self.scale), self.nu, np.log(self.tau)]
        res = minimize(nll, pars0, args = (Y,), method = 'BFGS', jac = grad_nll)
        return res.x

    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale,
                "nu": self.nu, "tau": self.tau)}

