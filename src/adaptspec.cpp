#include <RcppEigen.h>

#include "adaptspec.hpp"
#include "progress.hpp"

using namespace bayesspec;

// [[Rcpp::export(name=".adaptspec")]]
Rcpp::List adaptspec(
    unsigned int nLoop, unsigned int nWarmUp,
    unsigned int nSegmentsMax, Rcpp::NumericVector xR,
    unsigned int tMin,
    double sigmaSquaredAlpha,
    double tauPriorA,
    double tauPriorB,
    double tauUpperLimit,
    double probMM1,
    unsigned int nBases,
    unsigned int nSegmentsStart = 1,
    bool showProgress = false
) {
    RNG::initialise();

    Eigen::VectorXd x = Rcpp::as< Eigen::VectorXd >(xR);
    AdaptSpecPrior prior(
        nSegmentsMax,
        tMin,
        sigmaSquaredAlpha,
        tauPriorA, tauPriorB, tauUpperLimit,
        nBases
    );
    AdaptSpecSample start(x, prior, nSegmentsStart);
    AdaptSpecSampler sampler(x, start, probMM1, prior);

    unsigned int nSamples = nLoop - nWarmUp;
    Rcpp::IntegerVector nSegmentsSamples(nSamples);
    Rcpp::NumericVector betaSamples(Rcpp::Dimension({
        nSegmentsMax,
        1 + nBases,
        nSamples
    }));
    Rcpp::NumericMatrix tauSquaredSamples(nSegmentsMax, nSamples);
    Rcpp::IntegerMatrix cutPointsSamples(nSegmentsMax, nSamples);

    unsigned int nBetas = nSegmentsMax * (1 + nBases);

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
                sampler.getCurrent().tauSquared.data() + nSegmentsMax,
                tauSquaredSamples.begin() + sampleIndex * nSegmentsMax
            );
            std::copy(
                sampler.getCurrent().cutPoints.data(),
                sampler.getCurrent().cutPoints.data() + nSegmentsMax,
                cutPointsSamples.begin() + sampleIndex * nSegmentsMax
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
