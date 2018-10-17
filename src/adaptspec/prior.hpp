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
    unsigned int timeStep;
    bool cubeRoot;

    AdaptSpecPrior(
        unsigned int nSegmentsMin_,
        unsigned int nSegmentsMax_,
        unsigned int tMin_,
        double sigmaSquaredAlpha_,
        double tauPriorA_,
        double tauPriorB_,
        double tauUpperLimit_,
        unsigned int nBases_,
        unsigned int timeStep_,
        bool cubeRoot_
    ) : nSegmentsMin(nSegmentsMin_),
        nSegmentsMax(nSegmentsMax_),
        tMin(tMin_),
        sigmaSquaredAlpha(sigmaSquaredAlpha_),
        tauPriorA(tauPriorA_),
        tauPriorB(tauPriorB_),
        tauUpperLimit(tauUpperLimit_),
        nBases(nBases_),
        timeStep(timeStep_),
        cubeRoot(cubeRoot_) {}

    static AdaptSpecPrior fromList(const Rcpp::List& priorList) {
        return AdaptSpecPrior(
            priorList["n_segments_min"],
            priorList["n_segments_max"],
            priorList["t_min"],
            priorList["sigma_squared_alpha"],
            priorList["tau_prior_a"],
            priorList["tau_prior_b"],
            priorList["tau_upper_limit"],
            priorList["n_bases"],
            priorList["time_step"],
            priorList["cube_root"]
        );
    }

    static std::vector<AdaptSpecPrior> fromListOfLists(const Rcpp::List& priorsList) {
        std::vector<AdaptSpecPrior> priors;
        for (unsigned int i = 0; i < priorsList.size(); ++i) {
            priors.push_back(fromList(priorsList[i]));
        }
        return priors;
    }
};

}  // namespace bayespec

#endif  // SRC_ADAPTSPEC_PRIOR_HPP_
