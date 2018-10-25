#ifndef SRC_ADAPTSPEC_STATISTICS_HPP_
#define SRC_ADAPTSPEC_STATISTICS_HPP_

#include <RcppEigen.h>

namespace bayesspec {

class AdaptSpecStatistics {
public:
    AdaptSpecStatistics()
        : nCutpointWithin_(0),
          acceptedCutpointWithin_(0),
          nSingleWithin_(0),
          acceptedSingleWithin_(0),
          nHmcWithin_(0),
          acceptedHmcWithin_(0),
          nBetween_(0),
          acceptedBetween_(0) {}

    void acceptCutpointWithin() {
        ++nCutpointWithin_;
        ++acceptedCutpointWithin_;
    }

    void rejectCutpointWithin() {
        ++nCutpointWithin_;
    }

    void acceptSingleWithin() {
        ++nSingleWithin_;
        ++acceptedSingleWithin_;
    }

    void rejectSingleWithin() {
        ++nSingleWithin_;
    }

    void acceptBetween() {
        ++nBetween_;
        ++acceptedBetween_;
    }

    void rejectBetween() {
        ++nBetween_;
    }

    void acceptHmcWithin() {
        ++nHmcWithin_;
        ++acceptedHmcWithin_;
    }

    void rejectHmcWithin() {
        ++nHmcWithin_;
    }

    Rcpp::List asList() const {
        Rcpp::List output;
        output["n_cutpoint_within"] = nCutpointWithin_;
        output["accepted_cutpoint_within"] = acceptedCutpointWithin_;
        output["n_single_within"] = nSingleWithin_;
        output["accepted_single_within"] = acceptedSingleWithin_;
        output["n_hmc_within"] = nHmcWithin_;
        output["accepted_hmc_within"] = acceptedHmcWithin_;
        output["n_between"] = nBetween_;
        output["accepted_between"] = acceptedBetween_;
        return output;
    }

private:
    unsigned int nCutpointWithin_;
    unsigned int acceptedCutpointWithin_;

    unsigned int nSingleWithin_;
    unsigned int acceptedSingleWithin_;

    unsigned int nHmcWithin_;
    unsigned int acceptedHmcWithin_;

    unsigned int nBetween_;
    unsigned int acceptedBetween_;
};

}  // namespace bayesspec

#endif  // SRC_ADAPTSPEC_STATISTICS_HPP_
