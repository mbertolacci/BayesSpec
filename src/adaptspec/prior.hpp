#ifndef SRC_ADAPTSPEC_PRIOR_HPP_
#define SRC_ADAPTSPEC_PRIOR_HPP_

#include <RcppEigen.h>

namespace bayesspec {

struct AdaptSpecPrior {
    unsigned int nSegmentsMin;
    unsigned int nSegmentsMax;
    unsigned int tMin;
    double sigmaSquaredAlpha;
    double tauPriorA;
    double tauPriorB;
    double tauUpperLimit;
    unsigned int nBases;

    AdaptSpecPrior(
        unsigned int nSegmentsMin_,
        unsigned int nSegmentsMax_,
        unsigned int tMin_,
        double sigmaSquaredAlpha_,
        double tauPriorA_,
        double tauPriorB_,
        double tauUpperLimit_,
        unsigned int nBases_
    ) : nSegmentsMin(nSegmentsMin_),
        nSegmentsMax(nSegmentsMax_),
        tMin(tMin_),
        sigmaSquaredAlpha(sigmaSquaredAlpha_),
        tauPriorA(tauPriorA_),
        tauPriorB(tauPriorB_),
        tauUpperLimit(tauUpperLimit_),
        nBases(nBases_) {}

    static AdaptSpecPrior fromList(Rcpp::List priorList) {
        return AdaptSpecPrior(
            priorList["n_segments_min"],
            priorList["n_segments_max"],
            priorList["t_min"],
            priorList["sigma_squared_alpha"],
            priorList["tau_prior_a"],
            priorList["tau_prior_b"],
            priorList["tau_upper_limit"],
            priorList["n_bases"]
        );
    }
};

}  // namespace bayespec

#endif  // SRC_ADAPTSPEC_PRIOR_HPP_
