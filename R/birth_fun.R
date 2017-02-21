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
  for (k in 1:(nexp_prop-1)){
    if (k==1){
      log_prior_cut_prop = -log(nobs-(nexp_prop-k+1)*tmin+1)
    } else{
      log_prior_cut_prop = log_prior_cut_prop-log(nobs-xi_prop[k-1]-(nexp_prop-k+1)*tmin+1)
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
  for (k in 1:(nexp_curr-1)){
    if (k==1){
      log_prior_cut_curr = -log(nobs-(nexp_curr-k+1)*tmin+1)
    } else{
      log_prior_cut_curr = log_prior_cut_curr-log(nobs-xi_curr_temp[k-1]-(nexp_curr-k+1)*tmin+1)
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
