post_beta <- function(j, nseg_temp, x, xi_temp, tau_temp,
                      token_information){
  #   j=1 #used for debugging
  #   nseg_temp = nseg_temp[j] # used for debugging
  #   tau_temp = tau_temp[j] # used for debugging
  var_inflate <- token_information$var_inflate
  nbeta <- token_information$nbeta
  nbasis <- token_information$nbasis
  sigmasqalpha <- token_information$sigmasqalpha
  nfreq <- floor(nseg_temp/2)
  freq <- (0:nfreq)/(2*nfreq)
  if (j>1){
    dft <- fft(x[(xi_temp[j-1]+1):xi_temp[j]])/sqrt(nseg_temp)
    y <- dft[1:(nfreq+1)]
    prdgrm <- abs(y)^2
  } else {
    dft <- fft(x[1:xi_temp[j]])/sqrt(nseg_temp)
    y <- dft[1:(nfreq+1)]
    prdgrm <- abs(y)^2
  }
  nu_mat <- lin_basis_func(freq, nbeta)
  nn <- nseg_temp
  ytemp <- prdgrm
  param<-rep(0,nbeta)
  precs <- c(1/sigmasqalpha,rep(1/tau_temp,nbasis))
  opt <- trust(beta_derivs, param, rinit=1, rmax=100, parscale=rep(1,nbeta),
               iterlim = 100, fterm = sqrt(.Machine$double.eps),
               mterm = sqrt(.Machine$double.eps),
               minimize = FALSE, blather = FALSE, nn, nu_mat, prdgrm, precs)
  beta_mean <- opt$argument
  beta_var <- -solve(opt$hessian)
  list(beta_mean = beta_mean, beta_var = beta_var, nu_mat = nu_mat, prdgrm = prdgrm)
}
