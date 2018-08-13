from __future__ import division

import numpy as np
import pymc3 as pm
from scipy import stats
from theano import tensor as tt

if __name__ == '__main__':
    X = np.linspace(0, 50, 1000)
    Y = stats.norm(10, 2).pdf(X) * 10
    Y += np.random.normal(0, 0.1, len(X))
    E = np.ones_like(Y) * 0.1

    def gaussian_line(x, flux, width, centre):
        return flux * tt.exp(-((x - centre)**2) / 2 / width / width) / tt.sqrt(2 * np.pi) / width



    with pm.Model() as model:
        width = pm.HalfCauchy('width', beta=10)
        flux = pm.HalfNormal('flux', sd=10)
        centre = pm.Bound(pm.Normal, 0, 50)('centre', mu=25, sd=10)
        line = pm.Deterministic('line', gaussian_line(X, flux, width, centre))
        continuum = pm.Normal('continuum', mu=0, sd=10)

        spectrum = pm.Deterministic('spectrum', line+continuum)
        like = pm.Normal('like', mu=spectrum, sd=E, observed=Y)




        # print build_theano_function([width], model)
        # d = {i.name: type(i) for i in model.vars}
        # func = model.makefn([line])





    # plt.plot(X, Y, alpha=0.3)
    # plt.plot(X, out_y)
    # plt.show()