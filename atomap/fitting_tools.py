import scipy.odr
from math import sqrt
import numpy as np

def linear_fit_func(p, t):
    return p[0] * t + p[1]

def ODR_linear_fitter(x, y):
    """Orthogonal Distance Regresstion linear fitting.
    
    A least squares fitting method with perpendicular offsets. Often called
    Orthogonal Distance Regression (ODR). This method works better than
    Ordinary Least Squares fitting (OLS, with vertical offsets), on vertical
    lines.
    
    Parameters
    ----------
    x : array of x-values
    y : array of y-values
    
    Returns
    -------
    beta : array
        Array with the coefficients for y = ax + b.
        a = beta[0], b = beta[1]

    """
    Model = scipy.odr.Model(linear_fit_func)
    Data = scipy.odr.RealData(x, y)
    Odr = scipy.odr.ODR(Data, Model, [10000, 1], maxit = 10000)
    output = Odr.run()
    beta = output.beta
    betastd = output.sd_beta
    return(beta)
    
def get_shortest_distance_point_to_line(x_list,y_list,line):
    """Calculates the shortest distance from a point to a line given by a
    function.
    
    """
    x0, y0 = np.asarray(x_list), np.asarray(y_list)
    a, b, c = line[0], -1, line[1]
    num = a*x0 + b*y0 + c
    den = sqrt(a**2 + b**2)
    d = num/den
    return(d)
