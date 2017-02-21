whittle_like <- function(dd,fhat,n){
  #fhat is log spec density
  # A is the periodogram
  # dd=fit #debugging
  # n=nseg_curr_temp[j] #debugging
  A <- dd$prdgrm
  fhat <- as.vector(fhat)
  Ae <- as.vector(A*exp(-fhat))
  n1 <- floor(n/2)
  if (n %% 2 == 1){ #odd n
    f <- -sum(fhat[2:(n1+1)]+Ae[2:(n1+1)]) -.5*(fhat[1] + Ae[1]) -
      .5*n*log(2*pi)
  }else { # even n
    f <- -sum(fhat[2:n1]+Ae[2:n1]) -.5*(fhat[1] + Ae[1]) -
      .5*(fhat[n1+1] + Ae[n1+1]) - .5*n*log(2*pi)
  }
  return(f)
}
