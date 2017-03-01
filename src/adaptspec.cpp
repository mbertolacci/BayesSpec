#include <RcppEigen.h>

#include "adaptspec.hpp"
#include "progress.hpp"

using namespace bayesspec;

// [[Rcpp::export(name=".adaptspec")]]
Rcpp::List adaptspec(
    unsigned int nLoop,
    unsigned int nWarmUp,
    Rcpp::NumericMatrix xR,
    Rcpp::List priorList,
    double probMM1,
    unsigned int nSegmentsStart = 1,
    bool showProgress = false
) {
    RNG::initialise();

    Eigen::MatrixXd x = Rcpp::as< Eigen::MatrixXd >(xR);
    AdaptSpecPrior prior = AdaptSpecPrior::fromList(priorList);
    AdaptSpecSample start(x, prior, nSegmentsStart);
    AdaptSpecSampler sampler(x, start, probMM1, prior);

    unsigned int nSamples = nLoop - nWarmUp;
    Rcpp::IntegerVector nSegmentsSamples(nSamples);
    Rcpp::NumericVector betaSamples(Rcpp::Dimension({
        prior.nSegmentsMax,
        1 + prior.nBases,
        nSamples
    }));
    Rcpp::NumericMatrix tauSquaredSamples(prior.nSegmentsMax, nSamples);
    Rcpp::IntegerMatrix cutPointsSamples(prior.nSegmentsMax, nSamples);

    unsigned int nBetas = prior.nSegmentsMax * (1 + prior.nBases);

    ProgressBar progressBar(nLoop);
    for (unsigned int iteration = 0; iteration < nLoop; ++iteration) {
        sampler.sample();

        if (iteration % 100 == 0) {
            // NOTE(mgnb): checks whether the user has pressed Ctrl-C (among other things)
            Rcpp::checkUserInterrupt();
        }

        if (iteration >= nWarmUp) {
            unsigned int sampleIndex = iteration - nWarmUp;
            nSegmentsSamples[sampleIndex] = sampler.getCurrent().nSegments;
            std::copy(
                sampler.getCurrent().beta.data(),
                sampler.getCurrent().beta.data() + nBetas,
                betaSamples.begin() + sampleIndex * nBetas
            );
            std::copy(
                sampler.getCurrent().tauSquared.data(),
                sampler.getCurrent().tauSquared.data() + prior.nSegmentsMax,
                tauSquaredSamples.begin() + sampleIndex * prior.nSegmentsMax
            );
            std::copy(
                sampler.getCurrent().cutPoints.data(),
                sampler.getCurrent().cutPoints.data() + prior.nSegmentsMax,
                cutPointsSamples.begin() + sampleIndex * prior.nSegmentsMax
            );
        }

        if (showProgress) {
            ++progressBar;
        }
    }

    Rcpp::List result;
    result["n_segments"] = nSegmentsSamples;
    result["beta"] = betaSamples;
    result["tau_squared"] = tauSquaredSamples;
    result["cut_point"] = cutPointsSamples;

    return result;
}
