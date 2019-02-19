#ifndef SRC_ADAPTSPEC_SAMPLES_HPP_
#define SRC_ADAPTSPEC_SAMPLES_HPP_

#include <RcppEigen.h>
#include "parameters.hpp"
#include "prior.hpp"
#include "../samples.hpp"

namespace bayesspec {

class AdaptSpecSamples {
public:
    AdaptSpecSamples(
      unsigned int nSamples,
      unsigned int nSegmentsThin,
      unsigned int betaThin,
      unsigned int tauSquaredThin,
      unsigned int cutPointsThin,
      unsigned int muThin,
      const AdaptSpecPrior& prior
    ) : nSegments_(nSamples, nSegmentsThin),
        beta_(nSamples, betaThin, {
            prior.nSegmentsMax,
            1 + prior.nBases
        }),
        tauSquared_(nSamples, tauSquaredThin, prior.nSegmentsMax, true),
        cutPoints_(nSamples, cutPointsThin, prior.nSegmentsMax, true),
        mu_(nSamples, muThin, prior.nSegmentsMax, true) {}

    AdaptSpecSamples(
      unsigned int nSamples,
      const AdaptSpecPrior& prior
    ) : AdaptSpecSamples(nSamples, 1, 1, 1, 1, 1, prior) {}

    void save(const AdaptSpecParameters& parameters) {
        nSegments_.save(parameters.nSegments);
        beta_.save(parameters.beta);
        tauSquared_.save(parameters.tauSquared);
        cutPoints_.save(parameters.cutPoints);
        mu_.save(parameters.mu);
    }

    Rcpp::List asList() const {
        Rcpp::List output;
        output["n_segments"] = Rcpp::wrap(nSegments_);
        output["beta"] = Rcpp::wrap(beta_);
        output["tau_squared"] = Rcpp::wrap(tauSquared_);
        output["cut_points"] = Rcpp::wrap(cutPoints_);
        output["mu"] = Rcpp::wrap(mu_);
        return output;
    }

    static std::vector<AdaptSpecSamples> fromPriors(
        unsigned int nSamples,
        unsigned int nSegmentsThin,
        unsigned int betaThin,
        unsigned int tauSquaredThin,
        unsigned int cutPointsThin,
        unsigned int muThin,
        const std::vector<AdaptSpecPrior>& priors
    ) {
        std::vector<AdaptSpecSamples> output;
        for (unsigned int i = 0; i < priors.size(); ++i) {
            output.emplace_back(
                nSamples,
                nSegmentsThin,
                betaThin,
                tauSquaredThin,
                cutPointsThin,
                muThin,
                priors[i]
            );
        }
        return output;
    }

    static std::vector<AdaptSpecSamples> fromPriors(
        unsigned int nSamples,
        const std::vector<AdaptSpecPrior>& priors
    ) {
        std::vector<AdaptSpecSamples> output;
        for (unsigned int i = 0; i < priors.size(); ++i) {
            output.emplace_back(nSamples, priors[i]);
        }
        return output;
    }

private:
    Samples<unsigned int> nSegments_;
    Samples<double> beta_;
    Samples<double> tauSquared_;
    Samples<unsigned int> cutPoints_;
    Samples<double> mu_;
};

}  // namespace bayesspec

#endif  // SRC_ADAPTSPEC_SAMPLES_HPP_
