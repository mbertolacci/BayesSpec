# BayesSpec

## Overview

An implementation of methods for spectral analysis using the Bayesian framework. It includes functions for modelling spectrum as well as appropriate plotting and output estimates.

Please note that this is the development version of the R package BayesSpec. It is a substantial revision of the [existing package in CRAN](https://cran.r-project.org/package=BayesSpec).

## Installation

First, you need the dependency [acoda](https://github.com/mbertolacci/acoda/), a package that is not yet on CRAN

    devtools::install_github('mbertolacci/acoda')

Then you can get the latest development version from GitHub with

    devtools::install_github('mbertolacci/BayesSpec')

The previous version of the software can be installed from CRAN using

    install.packages('BayesSpec')

It is planned to update the CRAN version at some point, but there is currently no timeline for doing so.

This package runs a lot faster if the [FFTW](https://www.fftw.org/) package is installed. On macOS, this can be installed using [Homebrew](https://brew.sh/) and `brew install fftw`. On Ubuntu and other Debian-likes, it can be installed with `apt install libfftw-dev`. You will need to reinstall BayesSpec to make use of these routines.
