# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 11:21:20 2015

@author: gotzm
"""



import logging
from matplotlib.patches import Polygon
from numpy import pi
import numpy as np
from scipy.optimize import leastsq

__all__ = ["gauss2D", "fit_2D_gauss", "cross", "FitError"]

class FitError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def cross(x,y,wx,wy,color="r"):
    """matplotlib cross artist
    """
    marker = Polygon([[x,y],[x,y+wy],[x,y-wy],
                      [x,y],[x+wx,y],[x-wx,y]],
                     color=color)
    return marker

def gauss2D(amplitude, xCenter, yCenter, xWidth, yWidth=0, offset=0, rotation=0):
    """returns a 2D Gauss function with the given parameters
    
    Parameters
    --------------------
    amplitude : scalar
        the height of the function
    
    xCenter : scalar
        
    yCenter : scalar

    xWidth : scalar, optional
        sigma in the second direction (x)

    yWidth : scalar, optional
        sigma in the first direction (y), 0 makes it symmetric, yWidth = cWidth
    
    offset : scalar, optional
        minimum of the function, default is 0
    
    rotation : scalar
        counte-clockwise rotation angle in degrees
        
        
    Returns
    --------
    function(x, y) : a 2D Gaussian function
        because imshow calls the first index y and the second x, this function
        will apply xCenter and xWidth to the second argument and yCenter and 
        yWidth to the first. That way the fit function can work and return 
        as x-position and width the value one would expect when plotting the 
        data with imshow
        
    """
    #ensure float format
    amplitude=float(amplitude)
    xCenter = float(xCenter)
    yCenter = float(yCenter)
    xWidth = float(xWidth)
    #if no y width given, assume symmetric function    
    if yWidth == 0:
        yWidth = xWidth
    #rotate the coordinates around the center, for a rotation
    if rotation != 0:
        rota = pi/180.*rotation
        def function(y,x):
            xrot = (x-xCenter)*np.cos(rota)-(y-yCenter)*np.sin(rota)+xCenter
            yrot = (x-xCenter)*np.sin(rota)+(y-yCenter)*np.cos(rota)+yCenter
            return (offset+amplitude*np.exp(-0.5*(((xrot-xCenter)/xWidth)**2+
                                            ((yrot-yCenter)/yWidth)**2)))
                                            
    else:
        def function(y,x):
            return (offset+amplitude*np.exp(-0.5*(((x-xCenter)/xWidth)**2+
                                            ((y-yCenter)/yWidth)**2)))
    
    return function

def guess_2D_gauss(data):
    """Gives an initial guess for the parameters of a 2D Gaussian.
    
    Paramters
    -----------
    data : nd array
        the data to guess paramters for
        
    Returns
    ---------
    parameters : list
        contains amplitude, xCenter, yCenter, xWidth, yWidth, offset, rotation angle
        
    Calculation Details
    -------------
    center : x0 and y0 of the function
        calculated from the mean of the distribution
    width : sigma of the function
        calculated from the 2nd centered moment
    offset : 
        minimum of the data
        
    amplitude :
        maximum of data - minimum
        
    roation :
        always zero
    """
    total = data.sum()
    Y, X = np.indices(data.shape)
    yCenter = (Y*data).sum()/total
    xCenter = (X*data).sum()/total
    col = data[int(yCenter),:]
    xWidth = np.sqrt(((X[0]-xCenter)**2*col).sum()/col.sum())
    row = data[:,int(xCenter)]
    yWidth = np.sqrt(((Y[:,0]-yCenter)**2*row).sum()/row.sum())

    offset = np.min(data)
    amplitude = np.max(data)-offset

    return [amplitude, xCenter, yCenter, xWidth, yWidth, offset, 0.]    

def fit_2D_gauss(data,symmetric = False,useOffset=True, useRotation=True):
    """fit a 2D Gaussian on the provided data (minimize square difference)
    
    Parameters
    ------------
    data : nd array
        the data to be fitted
    symmetric : boolean, optional
        set to true if the fitted function should be symmetric, i.e. have equal
        sigma in x and y
    useOffse : boolean, optional
        set to False if the function should start at zero
    useRotation: boolean, optional
        set to False to not allow rotation, i.e. have the axis     parallel to the
        data axis
    
    Returns
    -----------
    popt : array
        the paramters returned by the fit, same order as gauss2D needs them
    cov : array
        covariance matrix of the fit as returned by scipy.optimize.leastsq
        
    Raises
    -------
    FitError
    
    """
    pGuess = guess_2D_gauss(data)

    #handle the different combinations of free parameters, remove the unwanted
    #parametrs from the guess and create appropriate objective functions
    if symmetric:
        if useOffset:
            pGuess.pop(6)
            pGuess.pop(4)
            errorFunc = lambda p: np.ravel(gauss2D(*p[0:4],offset=p[4])(*np.indices(data.shape))-data)
        else:
            pGuess = pGuess[0:4]
            errorFunc = lambda p: np.ravel(gauss2D(*p)(*np.indices(data.shape))-data)
    else:
        if not useRotation:
            pGuess.pop(6)
        if not useOffset:
            pGuess.pop(5)
            
        if not useOffset and useRotation:
            errorFunc = lambda p: np.ravel(gauss2D(*p[0:5],rotation=p[5])(*np.indices(data.shape))-data)
        else:
            errorFunc = lambda p: np.ravel(gauss2D(*p)(*np.indices(data.shape))-data)

    logging.debug("attempting fit with inital guess: "+str(pGuess))
    popt,cov_x,info,msg,success = leastsq(errorFunc, pGuess,full_output = True)
    
    if success != 1 and success != 2 and success != 3 and success !=4:
        raise FitError("Fit failed with message: "+msg)
    else:
        if cov_x is not None:
            cov = cov_x*np.square(errorFunc(popt)).sum()/(len(data)-len(pGuess))
        else:
            cov = np.inf
            raise FitError("none covariance matrix after {:d} iterations".format(info["nfev"]))
        return popt, cov