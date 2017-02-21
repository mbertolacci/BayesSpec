lin_basis_func <- function(freq, nbeta){
  nbasis <- nbeta-1
  n <- length(freq)
  omega <- matrix(0,n,nbasis)
  for (j in 1:nbasis){
    omega[,j] <- sqrt(2)*cos(2*j*pi*freq)/(2*pi*j)
  }
  return(cbind(rep(1,n),omega))
}
