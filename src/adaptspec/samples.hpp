#ifndef SRC_ADAPTSPEC_SAMPLES_HPP_
#define SRC_ADAPTSPEC_SAMPLES_HPP_

#include <RcppEigen.h>
#include "parameters.hpp"
#include "prior.hpp"

namespace bayesspec {

class AdaptSpecSamples {
public:
    AdaptSpecSamples(unsigned int nSamples, const AdaptSpecPrior& prior)
        : nSegments_(nSamples),
          beta_(Rcpp::Dimension({
              prior.nSegmentsMax,
              1 + prior.nBases,
              nSamples
          })),
          tauSquared_(prior.nSegmentsMax, nSamples),
          cutPoints_(prior.nSegmentsMax, nSamples),
          currentIndex_(0),
          nBetas_(prior.nSegmentsMax * (1 + prior.nBases)),
          nSegmentsMax_(prior.nSegmentsMax) {}

    void save(const AdaptSpecParameters& parameters) {
        nSegments_[currentIndex_] = parameters.nSegments;
        std::copy(
            parameters.beta.data(),
            parameters.beta.data() + nBetas_,
            beta_.begin() + currentIndex_ * nBetas_
        );
        std::copy(
            parameters.tauSquared.data(),
            parameters.tauSquared.data() + nSegmentsMax_,
            tauSquared_.begin() + currentIndex_ * nSegmentsMax_
        );
        std::copy(
            parameters.cutPoints.data(),
            parameters.cutPoints.data() + nSegmentsMax_,
            cutPoints_.begin() + currentIndex_ * nSegmentsMax_
        );

        ++currentIndex_;
    }

    Rcpp::List asList() const {
        Rcpp::List output;
        output["n_segments"] = nSegments_;
        output["beta"] = beta_;
        output["tau_squared"] = tauSquared_;
        output["cut_points"] = cutPoints_;
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
    Rcpp::IntegerVector nSegments_;
    Rcpp::NumericVector beta_;
    Rcpp::NumericMatrix tauSquared_;
    Rcpp::IntegerMatrix cutPoints_;

    unsigned int currentIndex_;

    unsigned int nBetas_;
    unsigned int nSegmentsMax_;
};

}  // namespace bayesspec

#endif  // SRC_ADAPTSPEC_SAMPLES_HPP_
