#ifndef SRC_PRIOR_HPP_
#define SRC_PRIOR_HPP_

namespace bayesspec {

struct AdaptSpecPrior {
    unsigned int nSegmentsMax;
    unsigned int tMin;
    double sigmaSquaredAlpha;
    double tauPriorA;
    double tauPriorB;
    double tauUpperLimit;
    unsigned int nBases;

    AdaptSpecPrior(
        unsigned int nSegmentsMax_,
        unsigned int tMin_,
        double sigmaSquaredAlpha_,
        double tauPriorA_,
        double tauPriorB_,
        double tauUpperLimit_,
        unsigned int nBases_
    ) : nSegmentsMax(nSegmentsMax_),
        tMin(tMin_),
        sigmaSquaredAlpha(sigmaSquaredAlpha_),
        tauPriorA(tauPriorA_),
        tauPriorB(tauPriorB_),
        tauUpperLimit(tauUpperLimit_),
        nBases(nBases_) {}
};

}  // namespace bayespec

#endif  // SRC_PRIOR_HPP_
