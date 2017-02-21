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
    for (k in 1:(nexp_temp-1)){
      if (k==1){
        log_prior_cut_prop <- -log(nobs-(nexp_temp-k+1)*tmin+1)
        log_prior_cut_curr <- -log(nobs-(nexp_temp-k+1)*tmin+1)
      } else {
        log_prior_cut_prop <- log_prior_cut_prop-log(nobs-xi_prop[k-1]-(nexp_temp-k+1)*tmin+1)
        log_prior_cut_curr <- log_prior_cut_curr-log(nobs-xi_curr_temp[k-1]-(nexp_temp-k+1)*tmin+1)
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
