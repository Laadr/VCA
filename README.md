# Vertex Component Analysis (VCA)

Translation in Python of the Vertex Component Analysis method to extract aset of endmembers (elementary spectrum) from a given hyperspectral image.

More details on the method:

Jose M. P. Nascimento and Jose M. B. Dias 

"Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004

# Usage
Ae, indice, Yp = vca(Y,R,verbose = True,snr_input = 0)

## Input variables
 Y - matrix with dimensions L(channels) x N(pixels)
     each pixel is a linear mixture of R endmembers
     signatures Y = M x s, where s = gamma x alfa
     gamma is a illumination perturbation factor and
     alfa are the abundance fractions of each endmember. NB: Y has to be a numpy array
     
 R - positive integer number of endmembers in the scene

## Output variables
Ae     - estimated mixing matrix (endmembers signatures)

indice - pixels that were chosen to be the most pure

Yp     - Data matrix Y projected.   

## Optional parameters
snr_input - (float) signal to noise ratio (dB)

verbose   - [True | False]

# Requirements

Scipy needs to be installed.

# Authors
Author: Adrien Lagrange (ad.lagrange@gmail.com)

This code is a translation of a matlab code provided by Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt) available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)

Translation of last version at 22-February-2018 (Matlab version 2.1 (7-May-2004))

Under Apache 2.0 license
