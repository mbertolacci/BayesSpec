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
  for (k in 1:(nexp_prop-1)){
    if (k==1){
      log_prior_cut_prop <- -log(nobs-(nexp_prop-k+1)*tmin+1)
    } else {
      log_prior_cut_prop <- log_prior_cut_prop-log(nobs-xi_prop[k-1]-(nexp_prop-k+1)*tmin+1)
    }
  }
  log_target_prop <- loglike_prop+log_tau_prior_prop+log_beta_prior_prop+log_prior_cut_prop
  #Evaluating Target density at current values
  log_prior_cut_curr <- 0
  for (k in 1:(nexp_curr_temp-1)){
    if (k==1) {
      log_prior_cut_curr <- -log(nobs-(nexp_curr_temp-k+1)*tmin+1)
    } else {
      log_prior_cut_curr <- log_prior_cut_curr-log(nobs-xi_curr_temp[k-1]-(nexp_curr_temp-k+1)*tmin+1)
    }
  }
  log_target_curr <- loglike_curr+log_prior_curr+log_prior_cut_curr
  met_rat <- min(1,exp(log_target_prop-log_target_curr+log_proposal_curr-log_proposal_prop+log_jacobian))
  list(met_rat = met_rat,nseg_prop = nseg_prop,xi_prop =xi_prop,tau_prop = tau_prop,
       beta_prop = beta_prop)
} # end function
