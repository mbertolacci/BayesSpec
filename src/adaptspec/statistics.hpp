#ifndef SRC_ADAPTSPEC_STATISTICS_HPP_
#define SRC_ADAPTSPEC_STATISTICS_HPP_

#include <RcppEigen.h>

namespace bayesspec {

class AdaptSpecStatistics {
public:
    AdaptSpecStatistics()
        : nWithin_(0),
          acceptedWithin_(0),
          nBetween_(0),
          acceptedBetween_(0),
          nHmc_(0),
          acceptedHmc_(0) {}

    void acceptWithin() {
        ++nWithin_;
        ++acceptedWithin_;
    }

    void rejectWithin() {
        ++nWithin_;
    }

    void acceptBetween() {
        ++nBetween_;
        ++acceptedBetween_;
    }

    void rejectBetween() {
        ++nBetween_;
    }

    void acceptHmc() {
        ++nHmc_;
        ++acceptedHmc_;
    }

    void rejectHmc() {
        ++nHmc_;
    }

    Rcpp::List asList() const {
        Rcpp::List output;
        output["n_within"] = nWithin_;
        output["accepted_within"] = acceptedWithin_;
        output["n_between"] = nBetween_;
        output["accepted_between"] = acceptedBetween_;
        output["n_hmc"] = nHmc_;
        output["accepted_hmc"] = acceptedHmc_;
        return output;
    }

private:
    unsigned int nWithin_;
    unsigned int acceptedWithin_;

    unsigned int nBetween_;
    unsigned int acceptedBetween_;

    unsigned int nHmc_;
    unsigned int acceptedHmc_;
};

}  // namespace bayesspec

#endif  // SRC_ADAPTSPEC_STATISTICS_HPP_
