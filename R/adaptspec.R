#' @name adaptspec
#'
#' @title Adaptive Spectral Estimation for Non-stationary Time Series
#'
#' @description Methodology for analyzing possibly non-stationary time series by adaptively dividing the time series into an unknown but finite number of segments and estimating the corresponding local spectra by smoothing splines.
#'
#' @param nloop The total number of MCMC iterations
#' @param nwarmup The number of burn-in iterations
#' @param nexp_max The maximum number of segments allowed
#' @param x The data, a univariate time series, not a time series object
#'
#' @param tmin The minimum number of observations per segment. An optional argument defaulted to tmin = 40.
#' @param sigmasqalpha An optional argument defaulted to sigmasqalpha = 100.
#' @param tau_prior_a An optional argurment defaulted to tau_prior_a = -1.
#' @param tau_prior_b An optional argurment defaulted to tau_prior_b = 0.
#' @param tau_up_limit An optional argurment defaulted to tau_up_limit = 10000.
#' @param prob_mm1 An optional argurment defaulted to prob_mm1 = 0.8.
#' @param step_size_max An optional argurment defaulted to step_size_max = 10.
#' @param var_inflate An optional argurment defaulted to var_inflate = 1.
#' @param nbasis An optional argurment defaulted to nbasis = 7.
#' @param nfreq_hat An optional argurment defaulted to nfreq_hat = 50.
#' @param plotting An optional argument for displaying output plots defaulted to FALSE. When set to TRUE, this displays the spectral and parition points.
#'
#' @return xi The partition points
#' @return log_spec_hat Estimates of the log spectra for all segments
#' @return nexp_curr The number of segments in each iteration.
#'
#' @import mvtnorm pscl trust
#'
#' @usage
#' adaptspec(nloop, nwarmup, nexp_max, x,
#'    tmin, sigmasqalpha, tau_prior_a, tau_prior_b,
#'    tau_up_limit, prob_mm1, step_size_max,
#'    var_inflate, nbasis, nfreq_hat, plotting)
#'
#' @examples
#' #Running adaptspec with the simulated_piecewise data.
#' data(simulated_piecewise)
#' model1 <- adaptspec(nloop = 2000, nwarmup = 500,
#'    nexp_max = 10, x = simulated_piecewise)
#' str(model1)
#' summary(model1$nexp_curr)
#' plot(model1$nexp_curr)
#' #Running adaptspec with a sample of the intracranial_eeg data and returing plots and summary.
#' data(intrcranial_eeg)
#' model2 <- adaptspec(nloop = 400, nwarmup = 100,
#'    nexp_max = 20, x = intracranial_eeg[1:2000], plotting = TRUE)
#' summary(model2)
#'
#' @author Rosen, O., Wood, S. and Stoffer, D.
#'
#' @references Rosen, O., Wood, S. and Stoffer, D. (2012). AdaptSPEC: Adaptive Spectral Estimation for Nonstationary Time Series. J. of the American Statistical Association, 107, 1575-1589
#'
#' @export
NULL

adaptspec <- function(nloop, nwarmup, nexp_max, x, tmin, sigmasqalpha, tau_prior_a, tau_prior_b, tau_up_limit, prob_mm1, step_size_max, var_inflate, nbasis, nfreq_hat, plotting){

#For optional variables,
if(missing(sigmasqalpha)){
  sigmasqalpha <- 100
} else{
  sigmasqalpha
}

if(missing(tau_prior_a)){
  tau_prior_a <- -1
} else{
  tau_prior_a
}

if(missing(tau_prior_b)){
  tau_prior_b <- 0
} else{
  tau_prior_b
}

if(missing(tau_up_limit)){
  tau_up_limit <- 10000
} else{
  tau_up_limit
}

if(missing(prob_mm1)){
  prob_mm1 <- 0.8
} else{
  prob_mm1
}

if(missing(step_size_max)){
  step_size_max <- 10
} else{
  step_size_max
}

if(missing(var_inflate)){
  var_inflate <- 1
} else{
  var_inflate
}

if(missing(nbasis)){
  nbasis <- 7
} else{
  nbasis
}

if(missing(nfreq_hat)){
  nfreq_hat <- 50
} else{
  nfreq_hat
}

if(missing(tmin)){
  tmin <- 40
} else{
  tmin
}

if(missing(plotting)){
  plotting <- FALSE
} else{
  plotting
}

if(plotting != TRUE){
  plotting <- FALSE
} else{
  plotting
}

lin_basis_func <- function(freq, nbeta){
  nbasis <- nbeta-1
  n <- length(freq)
  omega <- matrix(0,n,nbasis)
  for (j in 1:nbasis){
    omega[,j] <- sqrt(2)*cos(2*j*pi*freq)/(2*pi*j)
  }
  return(cbind(rep(1,n),omega))
}

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

within = function (x,nexp_temp,xi_curr_temp,beta_curr_temp,nseg_curr_temp,tau_temp,token_information){
  # nexp_temp = nexp_curr[p+1] #just for debugging
  xi_prop <- xi_curr_temp
  beta_prop <- beta_curr_temp
  nseg_new <- nseg_curr_temp
  nobs <- token_information$nobs
  nbeta <- token_information$nbeta
  nbasis <- token_information$nbasis
  sigmasqalpha <- token_information$sigmasqalpha
  prob_mm1 <- token_information$prob_mm1

  if (nexp_temp>1) {
    seg_temp <- sample(1:(nexp_temp-1),1,replace = TRUE) #Drawing Segment to relocate cutpoint
    u <- runif(1)
    cut_poss_curr <- xi_curr_temp[seg_temp]
    nposs_prior <- nseg_curr_temp[seg_temp]+nseg_curr_temp[seg_temp+1]-2*tmin+1
    if (u < prob_mm1){
      if (nseg_curr_temp[seg_temp] == tmin & nseg_curr_temp[seg_temp+1] == tmin) {
        nposs <- 1 # Number of possible locations for new cutpoint
        new_index<-sample(1:nposs, 1, replace = TRUE) #Drawing index of new cutpoint
        cut_poss_new <- xi_curr_temp[seg_temp]-1+new_index
      } else if (nseg_curr_temp[seg_temp] == tmin) {
        nposs <- 2 # Number of possible locations for new cutpoint
        new_index <- sample(1:nposs, 1, replace = TRUE) #Drawing index of new cutpoint
        cut_poss_new <- xi_curr_temp[seg_temp]-1+new_index
      } else if (nseg_curr_temp[seg_temp+1] == tmin) {
        nposs <- 2 # Number of possible locations for new cutpoint
        new_index <- sample(1:nposs,1,replace = TRUE) #Drawing index of new cutpoint
        cut_poss_new <- xi_curr_temp[seg_temp]+1-new_index
      } else {
        nposs <- 3 # Number of possible locations for new cutpoint
        new_index<-sample(1:nposs, 1, replace = TRUE) # Drawing index of new cutpoint
        cut_poss_new<-xi_curr_temp[seg_temp]-2+new_index
      }
    } else{ # u not < prob_mm1
      new_index <- sample(1:nposs_prior,1,replace = TRUE)
      if (seg_temp > 1){
        cut_poss_new <- sum(nseg_curr_temp[1:(seg_temp-1)])-1+tmin+new_index
      } else {
        cut_poss_new <- -1+tmin+new_index
      }
    }
    xi_prop[seg_temp] <- cut_poss_new
    if(seg_temp>1){
      nseg_new[seg_temp] <- xi_prop[seg_temp]-xi_curr_temp[seg_temp-1] #Number of observations in lower part of new cutpoint
    } else {
      nseg_new[seg_temp] <- xi_prop[seg_temp]
    }
    nseg_new[seg_temp+1] <- nseg_curr_temp[seg_temp]+nseg_curr_temp[seg_temp+1]-nseg_new[seg_temp] #Number of observations in upper part of new cutpoint
    # Evaluating the Proposal density for the cut-point at the cureent and proposed values
    if(abs(cut_poss_new-cut_poss_curr)>1){
      log_prop_cut_prop <- log(1-prob_mm1) - log(nposs_prior)
      log_prop_cut_curr <- log(1-prob_mm1) - log(nposs_prior)
    } else if (nseg_curr_temp[seg_temp] == tmin & nseg_curr_temp[seg_temp+1] == tmin){
      log_prop_cut_prop <- 0
      log_prop_cut_curr <- 0
    } else {
      if (nseg_curr_temp[seg_temp] == tmin || nseg_curr_temp[seg_temp+1] == tmin) {
        log_prop_cut_prop <- log(1-prob_mm1)-log(nposs_prior)+log(1/2)+log(prob_mm1)
      } else {
        log_prop_cut_prop <- log(1-prob_mm1)-log(nposs_prior)+log(1/3)+log(prob_mm1)
      }
      if (nseg_new[seg_temp] == tmin || nseg_new[seg_temp+1] == tmin) {
        log_prop_cut_curr <- log(1-prob_mm1)-log(nposs_prior)+log(1/2)+log(prob_mm1)
      } else {
        log_prop_cut_curr<-log(1-prob_mm1)-log(nposs_prior)+log(1/3)+log(prob_mm1)
      }
    }
    #Evaluating the Loglikelihood, Priors and Proposals at the
    #current values
    loglike_curr <- 0
    log_beta_curr_temp <- 0
    log_prior_curr <- 0
    for (j in seg_temp:(seg_temp+1)){
      fit <- post_beta(j,nseg_curr_temp[j],x,xi_curr_temp,tau_temp[j], token_information)
      #Compute log proposal density of beta at current  values
      log_beta_curr_temp <- log_beta_curr_temp+dmvnorm(beta_curr_temp[,j], fit$beta_mean, fit$beta_var, log = TRUE)
      fhat <- fit$nu_mat %*% beta_curr_temp[,j]
      #Compute Loglike at current values
      # cat("fhat:", fhat, "\n")
      log_curr_spec_dens <- whittle_like(fit, fhat,nseg_curr_temp[j])
      #       Ae = as.vector(fit$prdgrm*exp(-as.vector(fhat)))
      #       log_curr_spec_dens = -sum(fhat+Ae)
      # cat("log_curr_spec_dens:", log_curr_spec_dens, "\n")
      loglike_curr<-loglike_curr+log_curr_spec_dens
      #Compute priors at current values
      log_prior_curr <- log_prior_curr +
        dmvnorm(beta_curr_temp[,j], rep(0,nbeta), diag(c(sigmasqalpha,rep(tau_temp[j],nbasis))), log = TRUE)
    }
    #Evaluating the Loglikelihood, Priors and Proposals at the
    #proposed values Likelihood
    loglike_prop <- 0
    log_beta_prop <- 0
    log_prior_prop <- 0
    for (j in seg_temp:(seg_temp+1)){
      fit <- post_beta(j,nseg_new[j],x,xi_prop,tau_temp[j], token_information)
      beta_prop[,j]<-rmvnorm(1,fit$beta_mean,fit$beta_var)
      # Compute log proposal density of beta at proposed
      # values
      log_beta_prop <- log_beta_prop+dmvnorm(beta_prop[,j],fit$beta_mean, fit$beta_var, log = TRUE)
      fhat<-fit$nu_mat %*% beta_prop[,j]
      #Compute Loglike at proposed values
      log_prop_spec_dens <- whittle_like(fit,fhat,nseg_new[j])
      # cat("log_prop_spec_dens:", log_prop_spec_dens, "\n")
      loglike_prop <- loglike_prop + log_prop_spec_dens
      # Compute priors at proposed values
      log_prior_prop <- log_prior_prop +
        dmvnorm(beta_prop[,j], rep(0,nbeta), diag(c(sigmasqalpha,rep(tau_temp[j],nbasis))), log = TRUE)
    }
    #Proposal for beta
    log_proposal_curr <- log_beta_curr_temp+log_prop_cut_curr
    log_proposal_prop <- log_beta_prop+log_prop_cut_prop
    log_prior_cut_prop <- 0
    log_prior_cut_curr <- 0
    if (nexp_temp > 1) {
      for (k in 1:(nexp_temp-1)){
        if (k==1){
          log_prior_cut_prop <- -log(nobs-(nexp_temp-k+1)*tmin+1)
          log_prior_cut_curr <- -log(nobs-(nexp_temp-k+1)*tmin+1)
        } else {
          log_prior_cut_prop <- log_prior_cut_prop-log(nobs-xi_prop[k-1]-(nexp_temp-k+1)*tmin+1)
          log_prior_cut_curr <- log_prior_cut_curr-log(nobs-xi_curr_temp[k-1]-(nexp_temp-k+1)*tmin+1)
        }
      }
    }
    log_target_prop <- loglike_prop+log_prior_prop+log_prior_cut_prop
    log_target_curr <- loglike_curr+log_prior_curr+log_prior_cut_curr
  } else { #nexp_temp not greater than 1
    nseg_new <- nobs
    seg_temp <- 1
    fit <- post_beta(1,nobs,x,xi_prop,tau_temp, token_information)
    beta_prop <- t(rmvnorm(1,fit$beta_mean,fit$beta_var))
    #Compute log proposal density of beta at proposed  values
    log_beta_prop <- dmvnorm(as.vector(beta_prop),fit$beta_mean,fit$beta_var, log = TRUE)
    #Compute log proposal density of beta at current  values
    log_beta_curr_temp <- dmvnorm(beta_curr_temp,fit$beta_mean,fit$beta_var, log = TRUE)
    #Compute Loglike at proposed values
    fhat <- fit$nu_mat %*% beta_prop
    loglike_prop <- whittle_like(fit,fhat,nobs)
    # cat("loglike_prop:", loglike_prop,"\n")
    #Compute Loglike at current values
    fhat <- fit$nu_mat %*% beta_curr_temp
    loglike_curr <- whittle_like(fit,fhat,nobs)
    # cat("loglike_curr:", loglike_curr,"\n")
    #Compute Priors at proposed values
    log_prior_prop <- dmvnorm(t(beta_prop), rep(0,nbeta), diag(c(sigmasqalpha,rep(tau_temp,nbasis))), log = TRUE)
    #Compute Priors at current values
    log_prior_curr <- dmvnorm(beta_curr_temp, rep(0,nbeta), diag(c(sigmasqalpha,rep(tau_temp,nbasis))), log = TRUE)
    log_proposal_curr <- log_beta_curr_temp
    log_proposal_prop <- log_beta_prop
    log_target_prop <- loglike_prop+log_prior_prop
    log_target_curr <- loglike_curr+log_prior_curr
  } #end else {nexp_temp not greater than 1
  epsilon <- min(1,exp(log_target_prop-log_target_curr+log_proposal_curr-log_proposal_prop))
  list(epsilon = epsilon, xi_prop = xi_prop, beta_prop = beta_prop,
       nseg_new = nseg_new, seg_temp = seg_temp)
} #end func

death <- function(x,nexp_curr_temp,nexp_prop,tau_curr_temp,xi_curr_temp,nseg_curr_temp,
                  beta_curr_temp,log_move_curr,log_move_prop,token_information){

  # nexp_curr_temp <- nexp_curr[p] # just for debugging
  nobs <- token_information$nobs
  nbeta <- token_information$nbeta
  nbasis <- token_information$nbasis
  sigmasqalpha <- token_information$sigmasqalpha
  tmin <- token_information$tmin
  tau_up_limit <- token_information$tau_up_limit
  beta_prop <- matrix(0,nbeta,nexp_prop)
  tau_prop <- matrix(1,nexp_prop,1)
  nseg_prop <- matrix(0,nexp_prop,1)
  xi_prop <- matrix(0,nexp_prop,1)
  # Drawing  cut_point to delete
  cut_del <- sample(1:(nexp_curr_temp-1), 1, replace = TRUE)
  j <- 0
  for (k in 1:nexp_prop){
    j <- j+1
    if (k == cut_del){
      #*************************************************************
      # PROPOSED VALUES
      #*************************************************************
      xi_prop[k] <- xi_curr_temp[j+1]
      tau_prop[k] <- sqrt(tau_curr_temp[j]*tau_curr_temp[j+1]) # Combining 2 taus into 1
      nseg_prop[k] <- nseg_curr_temp[j]+nseg_curr_temp[j+1] # Combining two segments into 1
      #==================================================================
      # Evaluating the Likelihood, Proposal and Prior Densities at the Proposed values
      #==================================================================
      #Computing mean and variances for beta proposals
      fit <- post_beta(k,nseg_prop[k],x,xi_prop,tau_prop[k],token_information)
      beta_prop[,k] <- rmvnorm(1,fit$beta_mean,fit$beta_var)
      # Loglikelihood  at proposed values
      fhat <- fit$nu_mat %*% beta_prop[,k]
      # Ae = as.vector(fit$prdgrm*exp(-as.vector(fhat)))
      # log_prop_spec_dens = -sum(fhat+Ae)
      log_prop_spec_dens <- whittle_like(fit, fhat, nseg_prop[k])
      loglike_prop <- log_prop_spec_dens
      #==================================================================
      #Evaluating the Proposal Densities at the Proposed values of beta,
      #the cut points
      #==================================================================
      #Beta
      log_beta_prop <- dmvnorm(beta_prop[,k],fit$beta_mean, fit$beta_var, log = TRUE)
      #Segment
      log_seg_prop <- -log(nexp_curr_temp-1)
      # Calcualting Jacobian
      log_jacobian <- -log(2*(sqrt(tau_curr_temp[j])+sqrt(tau_curr_temp[j+1]))^2)
      #Calculating Log Proposal density at Proposed values
      log_proposal_prop <- log_beta_prop+log_seg_prop+log_move_prop
      #==================================================================
      #Evaluating the Prior Densities at the Proposed values for tau and beta
      #==============================================================
      # Beta
      log_beta_prior_prop <- dmvnorm(beta_prop[,k], rep(0,nbeta),
                                     diag(c(sigmasqalpha,rep(tau_prop[k],nbasis))), log = TRUE)
      # Tau
      log_tau_prior_prop <- -log(tau_up_limit)
      #*************************************************************
      # CURRENT VALUES
      #*************************************************************
      #=======================================
      # Evaluating the Likelihood, Proposal and Prior Densities at the Current values
      #=======================================
      #Beta Proposal and Prior
      log_beta_curr <- 0
      log_tau_prior_curr <- 0
      log_beta_prior_curr <- 0
      loglike_curr <- 0
      for (jj in j:(j+1)){
        fit <- post_beta(jj,nseg_curr_temp[jj],x,xi_curr_temp,tau_curr_temp[jj],token_information)
        log_beta_curr <- log_beta_curr+dmvnorm(beta_curr_temp[,jj],fit$beta_mean, fit$beta_var, log = TRUE)
        log_beta_prior_curr <- log_beta_prior_curr+dmvnorm(beta_curr_temp[,jj], rep(0,nbeta),
                                                           diag(c(sigmasqalpha,rep(tau_curr_temp[jj],nbasis))), log = TRUE)
        log_tau_prior_curr <- log_tau_prior_curr-log(tau_up_limit)
        # Loglikelihood  at Current values
        fhat <- fit$nu_mat %*% beta_curr_temp[,jj]
        # Ae = as.vector(fit$prdgrm*exp(-as.vector(fhat)))
        # log_curr_spec_dens = -sum(fhat+Ae)
        log_curr_spec_dens <- whittle_like(fit, fhat, nseg_curr_temp[jj])
        loglike_curr <- loglike_curr+log_curr_spec_dens
      }
      # Calculating Log proposal density at current values
      log_proposal_curr <- log_move_curr+log_beta_curr
      # Calculating Priors at Current Vlaues
      log_prior_curr <- log_beta_prior_curr+log_tau_prior_curr
      j <- j+1
    } else {
      xi_prop[k] <- xi_curr_temp[j]
      tau_prop[k] <- tau_curr_temp[j]
      nseg_prop[k] <- nseg_curr_temp[j]
      beta_prop[,k] <- beta_curr_temp[,j]
    } # end  if (k == cut_del)
  } # end for (k in 1:nexp_prop)
  #Evaluating Target density at proposed values
  log_prior_cut_prop <- 0
  if (nexp_prop > 1) {
    for (k in 1:(nexp_prop-1)){
      if (k==1){
        log_prior_cut_prop <- -log(nobs-(nexp_prop-k+1)*tmin+1)
      } else {
        log_prior_cut_prop <- log_prior_cut_prop-log(nobs-xi_prop[k-1]-(nexp_prop-k+1)*tmin+1)
      }
    }
  }
  log_target_prop <- loglike_prop+log_tau_prior_prop+log_beta_prior_prop+log_prior_cut_prop
  #Evaluating Target density at current values
  log_prior_cut_curr <- 0
  if (nexp_curr_temp > 1) {
    for (k in 1:(nexp_curr_temp-1)){
      if (k==1) {
        log_prior_cut_curr <- -log(nobs-(nexp_curr_temp-k+1)*tmin+1)
      } else {
        log_prior_cut_curr <- log_prior_cut_curr-log(nobs-xi_curr_temp[k-1]-(nexp_curr_temp-k+1)*tmin+1)
      }
    }
  }
  log_target_curr <- loglike_curr+log_prior_curr+log_prior_cut_curr
  met_rat <- min(1,exp(log_target_prop-log_target_curr+log_proposal_curr-log_proposal_prop+log_jacobian))
  list(met_rat = met_rat,nseg_prop = nseg_prop,xi_prop =xi_prop,tau_prop = tau_prop,
       beta_prop = beta_prop)
} # end function

birth <- function(x,nexp_curr,nexp_prop,tau_curr_temp,xi_curr_temp,nseg_curr_temp,beta_curr_temp,
                  log_move_curr,log_move_prop,token_information){
  # nexp_curr <- nexp_curr[p] # just for debugging
  nobs <- token_information$nobs
  nbeta <- token_information$nbeta
  nbasis <- token_information$nbasis
  sigmasqalpha <- token_information$sigmasqalpha
  tmin <- token_information$tmin
  tau_up_limit <- token_information$tau_up_limit
  beta_prop <- matrix(0,nbeta,nexp_prop)
  tau_prop <- matrix(1,nexp_prop,1)
  nseg_prop <- matrix(0,nexp_prop,1)
  xi_prop <- matrix(0,nexp_prop,1)
  #Drawing  segment to split
  kk <- which(nseg_curr_temp > 2*tmin) #Number of segments available for splitting
  nposs_seg <- length(kk)
  seg_cut <- kk[sample(1:nposs_seg, 1, replace = TRUE)] #Drawing segment to split
  nposs_cut <- nseg_curr_temp[seg_cut]-2*tmin+1 # Drawing New cutpoint
  for (jj in 1:nexp_curr){
    if (jj < seg_cut){
      xi_prop[jj] <- xi_curr_temp[jj]
      tau_prop[jj] <- tau_curr_temp[jj]
      nseg_prop[jj] <- nseg_curr_temp[jj]
      beta_prop[,jj] <- beta_curr_temp[,jj]
    } else if (jj == seg_cut){
      index <- sample(1:nposs_cut, 1, replace = TRUE)
      if (seg_cut==1){
        xi_prop[seg_cut] <- index+tmin-1
      } else{
        xi_prop[seg_cut] <- xi_curr_temp[jj-1]-1+tmin+index
      }
      xi_prop[seg_cut+1] <- xi_curr_temp[jj]
      zz <- runif(1) #Drawing new tausq
      tau_prop[seg_cut] <- tau_curr_temp[seg_cut]*zz/(1-zz)
      tau_prop[seg_cut+1] <- tau_curr_temp[seg_cut]*(1-zz)/zz
      nseg_prop[seg_cut] <- index+tmin-1
      nseg_prop[seg_cut+1] <- nseg_curr_temp[jj]-nseg_prop[seg_cut]
      for (k in jj:(jj+1)){
        fit <- post_beta(k,nseg_prop[k],x,xi_prop,tau_prop[k], token_information)
        beta_prop[,k] <- rmvnorm(1,fit$beta_mean,fit$beta_var) #Drawing a new value of beta
      }
    } else{
      xi_prop[jj+1] <- xi_curr_temp[jj]
      tau_prop[jj+1] <- tau_curr_temp[jj]
      nseg_prop[jj+1] <- nseg_curr_temp[jj]
      beta_prop[,jj+1] <- beta_curr_temp[,jj]
    } # end if (jj < seg_cut)
  } # end jj in 1:nexp_curr
  #Calculating Jacobian
  log_jacobian <- log(2*tau_curr_temp[seg_cut]/(zz*(1-zz)))
  #=======================================
  # Evaluating the Likelihood, Proposal and Prior Densities at the Proposed values
  #=======================================
  log_beta_prop <- 0
  log_tau_prior_prop <- 0
  log_beta_prior_prop <- 0
  loglike_prop <- 0
  for (jj in seg_cut:(seg_cut+1)){
    fit <- post_beta(jj,nseg_prop[jj],x,xi_prop,tau_prop[jj], token_information)
    log_beta_prop <- log_beta_prop+dmvnorm(beta_prop[,jj],fit$beta_mean, fit$beta_var, log = TRUE)
    log_beta_prior_prop <- log_beta_prior_prop+
      dmvnorm(beta_prop[,jj],rep(0,nbeta),diag(c(sigmasqalpha,rep(tau_prop[jj],nbasis))),
              log = TRUE )#Prior Density of beta
    log_tau_prior_prop <- log_tau_prior_prop-log(tau_up_limit) #Prior Density of Tausq
    fhat <- fit$nu_mat %*% beta_prop[,jj]
    # Ae = as.vector(fit$prdgrm*exp(-as.vector(fhat)))
    # log_prop_spec_dens = -sum(fhat+Ae)  #Loglikelihood  at proposed values
    log_prop_spec_dens <- whittle_like(fit, fhat, nseg_prop[jj])
    loglike_prop = loglike_prop + log_prop_spec_dens
  }
  log_seg_prop = -log(nposs_seg) #Proposal for Segment choice
  log_cut_prop = -log(nposs_cut) #Proposal for Cut point choice
  #Evaluating prior density for cut points at proposed values
  log_prior_cut_prop = 0
  if (nexp_prop > 1) {
    for (k in 1:(nexp_prop-1)){
      if (k==1){
        log_prior_cut_prop = -log(nobs-(nexp_prop-k+1)*tmin+1)
      } else{
        log_prior_cut_prop = log_prior_cut_prop-log(nobs-xi_prop[k-1]-(nexp_prop-k+1)*tmin+1)
      }
    }
  }
  #Calculating Log Proposal density at Proposed values
  log_proposal_prop = log_beta_prop+log_seg_prop+log_move_prop+log_cut_prop
  #Calculating Log Prior density at Proposed values
  log_prior_prop = log_beta_prior_prop+log_tau_prior_prop+log_prior_cut_prop
  #Calculating Target density at Proposed values
  log_target_prop = loglike_prop+log_prior_prop
  #*************************************************************
  #CURRENT VALUES
  #*************************************************************
  #=======================================
  #Evaluating the Likelihood, Proposal and Prior Densities at the Current values
  #=======================================
  #Beta Proposal and Prior
  fit = post_beta(seg_cut,nseg_curr_temp[seg_cut],x,xi_curr_temp,tau_curr_temp[seg_cut],token_information)
  if (nexp_curr ==1) {
    log_beta_curr = dmvnorm(beta_curr_temp,fit$beta_mean,fit$beta_var,log=TRUE)
    log_beta_prior_curr = dmvnorm(beta_curr_temp,rep(0,nbeta),
                                  diag(c(sigmasqalpha,rep(tau_curr_temp[seg_cut],nbasis))),log=TRUE)
  }else{
    log_beta_curr = dmvnorm(beta_curr_temp[,seg_cut],fit$beta_mean,fit$beta_var,log=TRUE)
    log_beta_prior_curr = dmvnorm(beta_curr_temp[,seg_cut],
                                  rep(0,nbeta),diag(c(sigmasqalpha,rep(tau_curr_temp[seg_cut],nbasis))),log=TRUE)
  }
  log_tau_prior_curr = -log(tau_up_limit)
  #Loglikelihood  at current values
  if (nexp_curr ==1){
    fhat = fit$nu_mat %*% beta_curr_temp
  } else{
    fhat = fit$nu_mat %*% beta_curr_temp[,seg_cut]
  }
  # Ae = as.vector(fit$prdgrm*exp(-as.vector(fhat)))
  # log_curr_spec_dens = -sum(fhat+Ae)
  log_curr_spec_dens = whittle_like(fit, fhat, nseg_curr_temp[seg_cut])
  loglike_curr = log_curr_spec_dens
  #Calculating Log proposal density at current values
  log_proposal_curr = log_beta_curr+log_move_curr
  #Evaluating  prior density for cut points at current values
  log_prior_cut_curr = 0
  if (nexp_curr > 1) {
    for (k in 1:(nexp_curr-1)){
      if (k==1){
        log_prior_cut_curr = -log(nobs-(nexp_curr-k+1)*tmin+1)
      } else{
        log_prior_cut_curr = log_prior_cut_curr-log(nobs-xi_curr_temp[k-1]-(nexp_curr-k+1)*tmin+1)
      }
    }
  }
  if (nexp_curr == 1) log_prior_cut_curr = 0
  #Calculating Priors at Current Vlaues
  log_prior_curr = log_beta_prior_curr+log_tau_prior_curr+log_prior_cut_curr
  #Evalulating Target densities at current values
  log_target_curr = loglike_curr+log_prior_curr
  met_rat = min(1,exp(log_target_prop-log_target_curr+log_proposal_curr-log_proposal_prop+log_jacobian))
  list(met_rat = met_rat,nseg_prop = nseg_prop,xi_prop = xi_prop,tau_prop = tau_prop,
       beta_prop = beta_prop)
} # end function

x0 <- 1:length(x)
x <- lm(x ~ x0)$res
nobs <- length(x)
tt <- 1:nobs
nbeta <- nbasis+1
token_information <- list(nobs = nobs, var_inflate = var_inflate,
                          nbasis = nbasis, nbeta = nbeta,
                          sigmasqalpha= sigmasqalpha,
                          prob_mm1= prob_mm1, tmin = tmin,
                          tau_prior_a = tau_prior_a,
                          tau_prior_b = tau_prior_b,
                          tau_up_limit = tau_up_limit,
                          step_size_max = step_size_max)
tausq <- matrix(list(), nexp_max, 1)
beta <- matrix(list(), nexp_max,1)
xi <- matrix(list(), nexp_max,1) #Cutpoint locations xi_1 is first cutpoint, xi_) is beginning of timeseries
nseg <- matrix(list(), nexp_max,1) #Number of observations in each segment
log_spec_hat <- matrix(list(), nexp_max,1)

freq_hat <- (0:nfreq_hat)/(2*nfreq_hat)
nu_mat_hat <- lin_basis_func(freq_hat, nbeta)
for (j in 1:nexp_max){
  tausq[[j]] <- matrix(1, j, nloop+1)
  beta[[j]] <- array(rep(1,nbeta*j*(nloop+1)), dim = c(nbeta,j,nloop+1))
  xi[[j]] <- matrix(1,j,nloop+1)
  nseg[[j]] <- matrix(1,j,nloop+1)
  log_spec_hat[[j]] <- array(rep(0,(nfreq_hat+1)*j*(nloop+1)), dim = c(nfreq_hat+1,j,nloop+1))
}
nexp_curr <- rep(1, nloop+1)
for (j in 1:nexp_curr[1]){
  tausq[[nexp_curr[1]]][j,1] <- runif(1)*tau_up_limit
}

for (j in 1:nexp_curr[1]){
  if (nexp_curr[1]==1){
    xi[[nexp_curr[1]]][j,1] <- nobs
    nseg[[nexp_curr[1]]][j,1] <- nobs
  } else {
    if (j==1){
      nposs <- nobs - nexp_curr[1]*tmin+1
      xi[[nexp_curr[1]]][j,1] <- tmin+sample(1:nposs,1,replace = TRUE)-1
      nseg[[nexp_curr[1]]][j,1] <- xi[[nexp_curr[1]]][j,1]
    } else if (j>1 & j < nexp_curr[1]){
      nposs <- nobs-xi[[nexp_curr[1]]][j-1,1]-tmin*(nexp_curr[1]-j+1)+1
      xi[[nexp_curr[1]]][j,1] <- tmin+sample(1:nposs,1,replace = TRUE)+xi[[nexp_curr[1]]][j-1,1]-1
      nseg[[nexp_curr[1]]][j,1] <- xi[[nexp_curr[1]]][j,1]-xi[[nexp_curr[1]]][j-1,1]
    } else {
      xi[[nexp_curr[1]]][j,1] <- nobs
      nseg[[nexp_curr[1]]][j,1] <- xi[[nexp_curr[1]]][j,1] - xi[[nexp_curr[1]]][j-1,1]
    }
  }
}
xi_temp <- xi[[nexp_curr[1]]][,1]
nseg_temp <- nseg[[nexp_curr[1]]][,1]
tau_temp <- tausq[[nexp_curr[1]]][,1]
for (j in 1:nexp_curr[1]){
  fit <- post_beta(j,nseg_temp[j],x,xi_temp,tau_temp[j], token_information)
  beta[[nexp_curr[1]]][,j,1] <- as.vector(rmvnorm(1, fit$beta_mean, fit$beta_var))
}
epsilon <- rep(0, nloop)
met_rat <- rep(0, nloop)
Rev_Jump <- 1

for (p in 1:nloop){
  if(p %% 100 == 0){
    cat("p:", p, " \n")
  }
  if (Rev_Jump == 1){
    # BETWEEN MODEL MOVE
    # Number of available segments
    kk <- length(which(nseg[[nexp_curr[p]]][,p] > 2*tmin))
    # Deciding on birth or death
    if (kk == 0){ #Stay where you (if nexp_curr=1) or join segments if there are no available segments to cut
      if(nexp_curr[p] == 1){
        nexp_prop <- nexp_curr[p] # Stay where you are
        log_move_prop <- 0
        log_move_curr <- 0
      }else{
        nexp_prop <- nexp_curr[p] - 1 # join segments
        log_move_prop <- 1
        if (nexp_prop == 1){
          log_move_curr <- 1
        }else{
          log_move_curr <- log(0.5)
        }
      }
    } else{ # kk is not == 0
      if (nexp_curr[p] == 1){
        nexp_prop <- nexp_curr[p]+1
        log_move_prop <- 0
        if (nexp_prop == nexp_max){
          log_move_curr <- 0
        } else{
          log_move_curr <- log(0.5)
        }
      } else if (nexp_curr[p] == nexp_max){
        nexp_prop <- nexp_curr[p]-1
        log_move_prop <- 0
        if(nexp_prop == 1){
          log_move_curr <- 0
        } else{
          log_move_curr <- log(0.5)
        }
      } else{
        u <- runif(1)
        if (u < .5){
          nexp_prop <- nexp_curr[p]+1
          if (nexp_prop==nexp_max) {
            log_move_curr <- 0
            log_move_prop <- log(0.5)
          } else {
            log_move_curr <- log(0.5)
            log_move_prop <- log(0.5)
          }
        } else{
          nexp_prop <- nexp_curr[p]-1
          if (nexp_prop == 1){
            log_move_curr <- 0
            log_move_prop <- log(0.5)
          } else{
            log_move_curr <- log(0.5)
            log_move_prop <- log(0.5)
          }
        }
      }
    } # end if (kk = 0)
    xi_curr_temp <- xi[[nexp_curr[p]]][,p]
    beta_curr_temp <- beta[[nexp_curr[p]]][,,p]
    tau_curr_temp <- tausq[[nexp_curr[p]]][,p]
    nseg_curr_temp <- nseg[[nexp_curr[p]]][,p]
    if (nexp_prop < nexp_curr[p]){
      #Death
      death_step <- death(x,nexp_curr[p],nexp_prop,tau_curr_temp,xi_curr_temp,nseg_curr_temp,
                          beta_curr_temp,log_move_curr,log_move_prop, token_information)
      met_rat[p] <- death_step$met_rat
      nseg_prop <- death_step$nseg_prop
      xi_prop <- death_step$xi_prop
      tau_prop <- death_step$tau_prop
      beta_prop <- death_step$beta_prop
    } else if (nexp_prop > nexp_curr[p]) {
      # Birth
      birth_step <- birth(x,nexp_curr[p],nexp_prop,tau_curr_temp,
                          xi_curr_temp,nseg_curr_temp,beta_curr_temp,
                          log_move_curr,log_move_prop,token_information)
      met_rat[p] <- birth_step$met_rat
      nseg_prop <- birth_step$nseg_prop
      xi_prop <- birth_step$xi_prop
      tau_prop <- birth_step$tau_prop
      beta_prop <- birth_step$beta_prop
    } else {
      xi_prop <- xi[[nexp_curr[p]]][,p]
      nseg_prop <- nseg[[nexp_curr[p]]][,p]
      tau_prop <- tausq[[nexp_curr[p]]][,p]
      beta_prop <- beta[[nexp_curr[p]]][,,p]
      met_rat[p] <- 1
    }
    u <- runif(1)
    if (u < met_rat[p]){
      nexp_curr[p+1] <- nexp_prop
      xi[[nexp_curr[p+1]]][,p+1] <- xi_prop
      nseg[[nexp_curr[p+1]]][,p+1] <- nseg_prop
      tausq[[nexp_curr[p+1]]][,p+1] <- tau_prop
      beta[[nexp_curr[p+1]]][,,p+1] <- beta_prop
    } else{
      nexp_curr[p+1] <- nexp_curr[p]
      xi[[nexp_curr[p+1]]][,p+1] <- xi[[nexp_curr[p+1]]][,p]
      nseg[[nexp_curr[p+1]]][,p+1] <- nseg[[nexp_curr[p+1]]][,p]
      tausq[[nexp_curr[p+1]]][,p+1] <- tausq[[nexp_curr[p+1]]][,p]
      beta[[nexp_curr[p+1]]][,,p+1] <- beta[[nexp_curr[p+1]]][,,p]
    }
  } else { # Rev_Jump = 0
    nexp_curr[p+1] <- nexp_curr[p]
    xi[[nexp_curr[p+1]]][,p+1] <- xi[[nexp_curr[p+1]]][,p]
    nseg[[nexp_curr[p+1]]][,p+1] <- nseg[[nexp_curr[p+1]]][,p]
    tausq[[nexp_curr[p+1]]][,p+1] <- tausq[[nexp_curr[p+1]]][,p]
    beta[[nexp_curr[p+1]]][,,p+1] <- beta[[nexp_curr[p+1]]][,,p]
  }
  #WITHIN MODEL MOVE
  #Drawing a new cut point and beta simultaneously
  #First draw the size of the move
  xi_curr_temp <- xi[[nexp_curr[p+1]]][,p+1]
  beta_curr_temp <- beta[[nexp_curr[p+1]]][,,p+1]
  tau_temp <- tausq[[nexp_curr[p+1]]][,p+1]
  nseg_curr_temp <- nseg[[nexp_curr[p+1]]][,p+1]
  within_step <- within(x,nexp_curr[p+1],xi_curr_temp,beta_curr_temp,
                        nseg_curr_temp,tau_temp,token_information)
  epsilon[p] <- within_step$epsilon
  xi_prop <- within_step$xi_prop
  beta_prop <- within_step$beta_prop
  nseg_new <- within_step$nseg_new
  seg_temp <- within_step$seg_temp
  u <- runif(1)
  if (u < epsilon[p] || p==1){
    if (nexp_curr[p+1] > 1){
      for (j in seg_temp:(seg_temp+1)){
        beta[[nexp_curr[p+1]]][,j,p+1] <- beta_prop[,j]
        xi[[nexp_curr[p+1]]][j,p+1] <- xi_prop[j]
        nseg[[nexp_curr[p+1]]][j,p+1] <- nseg_new[j]
      }
    } else{
      beta[[nexp_curr[p+1]]][,1,p+1] <- beta_prop[,1]
    }
  } else {
    beta[[nexp_curr[p+1]]][,,p+1] <- beta_curr_temp
    xi[[nexp_curr[p+1]]][,p+1] <- xi_curr_temp
    nseg[[nexp_curr[p+1]]][,p+1] <- nseg_curr_temp
  }
  # Drawing tausq
  for (j in 1:nexp_curr[p+1]){
    tau_a <- nbasis/2 + tau_prior_a
    tau_b <- sum(beta[[nexp_curr[p+1]]][2:nbeta,j,p+1]^2)/2+tau_prior_b
    u <- runif(1)
    const1 <- pigamma(tau_up_limit, tau_a, tau_b)
    const2 <- u*const1
    tausq[[nexp_curr[p+1]]][j,p+1] <- qigamma(const2, tau_a, tau_b)
  }
  # Estimating Spectral Density
  for (j in 1:nexp_curr[p+1]){
    log_spec_hat[[nexp_curr[p+1]]][,j,p+1] <- nu_mat_hat %*% beta[[nexp_curr[p+1]]][,j,p+1]
  }
} # end gibbs loop
fmean <- matrix(list(), nexp_max, 1)
for (j in 1:nexp_max){
  fmean[[j]] <- matrix(1,length(freq_hat),j)
}

if (plotting == TRUE){

  #Plots of individual Spectra
  for (j in 1:nexp_max){
    kk <- which(nexp_curr[(nwarmup+1):nloop] == j)
    if (length(kk) != 0){
      fmean[[j]] <- apply(log_spec_hat[[j]][,,kk+nwarmup],c(1,2), mean)
      for (k in 1:j){
        plot(freq_hat,fmean[[j]][,k], type = "l", main = paste("Log Spectral Density for component number", k, " in a mixture of", j))
        # lines(true_spec_dens[[k]]$freq,log(true_spec_dens[[k]]$spec),type = "l", col = "red")
      }
    }
  }

  # Plots of Partition Points
  for (j in 1:nexp_max){
    kk <- which(nexp_curr[(nwarmup+1):nloop] == j)
    if (length(kk) != 0 & j > 1){
      for (k in 1:(j-1)){
        plot(xi[[j]][k,kk+nwarmup], type = "l", main = paste("Plot of",k,"th partition points"))
      }
      for (k in 1:(j-1)){
        hist(xi[[j]][k,kk+nwarmup])
      }
    }
  }
}

z <- list(xi = xi,
          beta = beta,
          log_spec_hat = log_spec_hat,
          nloop = nloop,
          nwarmup = nwarmup,
          nexp_max = nexp_max,
          x = x,
          nexp_curr = nexp_curr,
          nexp_max = nexp_max,
          tmin = tmin,
          sigmasqalpha = sigmasqalpha,
          tau_prior_a = tau_prior_a,
          tau_prior_b = tau_prior_b,
          tau_up_limit = tau_up_limit,
          prob_mm1 = prob_mm1,
          step_size_max = step_size_max,
          var_inflate = var_inflate,
          nbasis = nbasis,
          nfreq_hat = nfreq_hat)
return(z)
}
