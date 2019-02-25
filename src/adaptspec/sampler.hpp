#ifndef SRC_ADAPTSPEC_SAMPLER_HPP_
#define SRC_ADAPTSPEC_SAMPLER_HPP_

#include <RcppEigen.h>

#include "prior.hpp"
#include "state.hpp"

namespace bayesspec {

class AdaptSpecSampler {
public:
    AdaptSpecSampler(
        Eigen::MatrixXd& x,
        const std::vector<Eigen::VectorXi>& missingIndices,
        const AdaptSpecParameters& start,
        const AdaptSpecTuning& tuning,
        const AdaptSpecPrior& prior
    ) : current_(start, x, missingIndices, prior, tuning) {}

    template<typename RNG>
    void sample(RNG& rng) {
        current_.sample(rng);
    }

    void endWarmUp() {
        current_.endWarmUp();
    }

    const AdaptSpecParameters& getCurrent() const {
        return current_.parameters;
    }

    double getLogPosterior() const {
        return current_.getLogPosterior();
    }

    const AdaptSpecStatistics& getWarmUpStatistics() const {
        return current_.getWarmUpStatistics();
    }

    const AdaptSpecStatistics& getStatistics() const {
        return current_.getStatistics();
    }

    void debugOutput() const {
        Rcpp::Rcout << current_;
    }

private:
    AdaptSpecState current_;
};

}  // namespace bayespec

#endif  // SRC_ADAPTSPEC_SAMPLER_HPP_
