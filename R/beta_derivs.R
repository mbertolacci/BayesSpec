beta_derivs <- function(param,n,xx,A,precs){
  n1 <- floor(n/2)
  xxb <- xx %*% param
  Ae <- as.vector(A*exp(-xxb))
  if (n %% 2 == 1){ #odd n
    f <- -sum(xxb[2:(n1+1)]+Ae[2:(n1+1)]) -.5*(xxb[1] + Ae[1]) -
      .5*t(param) %*% diag(precs) %*% param
    g <- -t(xx[2:(n1+1),]) %*% (1-Ae[2:(n1+1)]) -
      .5*(1-Ae[1])*xx[1,] - diag(precs) %*% param
    h <- -t(xx[2:(n1+1),]) %*% diag(Ae[2:(n1+1)]) %*% xx[2:(n1+1),]
    -.5*Ae[1] * xx[1,] %*% t(xx[1,]) - diag(precs)
  }else { # even n
    f <- -sum(xxb[2:n1]+Ae[2:n1]) -.5*(xxb[1] + Ae[1]) -
      .5*(xxb[n1+1] + Ae[n1+1]) - .5*t(param) %*% diag(precs) %*% param
    g <- -t(xx[2:n1,]) %*% (1-Ae[2:n1]) - .5*(1-Ae[1])*xx[1,] -
      .5*(1-Ae[n1+1])*xx[n1+1,] - diag(precs) %*% param
    h <- -t(xx[2:n1,]) %*% diag(Ae[2:n1]) %*% xx[2:n1,] -.5*Ae[1] * xx[1,] %*% t(xx[1,]) -
      .5*Ae[n1+1] * xx[n1+1,] %*% t(xx[n1+1,]) - diag(precs)
  }
  list(value = f, gradient = g, hessian = h)
}
