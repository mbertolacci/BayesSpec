#ifndef SRC_ADAPTSPEC_PRIOR_HPP_
#define SRC_ADAPTSPEC_PRIOR_HPP_

#include <RcppEigen.h>

namespace bayesspec {

struct AdaptSpecPrior {
    enum FrequencyTransform {
        IDENTITY,
        CUBE_ROOT
    };

    unsigned int nSegmentsMin;
    unsigned int nSegmentsMax;
    unsigned int tMin;
    double sigmaSquaredAlpha;
    double tauPriorA;
    double tauPriorB;
    double tauUpperLimit;
    unsigned int nBases;
    unsigned int timeStep;
    FrequencyTransform frequencyTransform;
    bool segmentMeans;
    double muLower;
    double muUpper;

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
        FrequencyTransform frequencyTransform_,
        bool segmentMeans_,
        double muLower_,
        double muUpper_
    ) : nSegmentsMin(nSegmentsMin_),
        nSegmentsMax(nSegmentsMax_),
        tMin(tMin_),
        sigmaSquaredAlpha(sigmaSquaredAlpha_),
        tauPriorA(tauPriorA_),
        tauPriorB(tauPriorB_),
        tauUpperLimit(tauUpperLimit_),
        nBases(nBases_),
        timeStep(timeStep_),
        frequencyTransform(frequencyTransform_),
        segmentMeans(segmentMeans_),
        muLower(muLower_),
        muUpper(muUpper_) {}

    static AdaptSpecPrior fromList(const Rcpp::List& priorList) {
        std::string frequencyTransformName = priorList["frequency_transform"];
        FrequencyTransform frequencyTransform = AdaptSpecPrior::IDENTITY;
        if (frequencyTransformName == "cbrt") {
            frequencyTransform = AdaptSpecPrior::CUBE_ROOT;
        }
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
            frequencyTransform,
            priorList["segment_means"],
            priorList["mu_lower"],
            priorList["mu_upper"]
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
