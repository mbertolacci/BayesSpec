#include <RcppEigen.h>
#include "splines.hpp"

using namespace bayesspec;

using Eigen::VectorXd;

using Rcpp::IntegerVector;
using Rcpp::IntegerMatrix;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

// [[Rcpp::export(name=".time_varying_spectra_samples")]]
NumericVector timeVaryingSpectraSamples(
    IntegerVector nSegments,
    IntegerMatrix cutPoints,
    NumericVector beta,
    unsigned int nFrequencies
) {
    const NumericVector& betaDims = beta.attr("dim");

    unsigned int nIterations = nSegments.size();
    unsigned int nSegmentsMax = cutPoints.ncol();
    unsigned int nTimes = *std::max_element(cutPoints.begin(), cutPoints.end());
    unsigned int nBeta = betaDims[2];
    unsigned int nBases = nBeta - 1;

    VectorXd frequencies = VectorXd::LinSpaced(
        nFrequencies, 0, nFrequencies - 1
    ) / static_cast<double>(2 * (nFrequencies - 1));
    Eigen::MatrixXd nuHat = splineBasis1dDemmlerReinsch(frequencies, nBases);
    VectorXd segmentBeta(nBeta);

    NumericVector output(Rcpp::Dimension({ nIterations, nFrequencies, nTimes }));
    for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
        unsigned int segmentStart = 0;
        unsigned int iterationNSegments = static_cast<unsigned int>(nSegments[iteration]);
        for (unsigned int segment = 0; segment < iterationNSegments; ++segment) {
            for (unsigned int i = 0; i < nBeta; ++i) {
                segmentBeta[i] = beta[
                    i * nSegmentsMax * nIterations
                    + segment * nIterations
                    + iteration
                ];
            }
            VectorXd segmentSpectra = nuHat * segmentBeta;

            unsigned int maxTime = static_cast<unsigned int>(cutPoints(iteration, segment));
            for (unsigned int time = segmentStart; time < maxTime; ++time) {
                for (unsigned int freq = 0; freq < nFrequencies; ++freq) {
                    output[
                        time * nFrequencies * nIterations
                        + freq * nIterations
                        + iteration
                    ] = segmentSpectra[freq];
                }
            }

            segmentStart = cutPoints(iteration, segment);
        }
    }
    output.attr("frequencies") = Rcpp::wrap(frequencies);

    return output;
}

// [[Rcpp::export(name=".time_varying_spectra_mixture_mean")]]
NumericVector timeVaryingSpectraMixtureMean(NumericVector componentSamples, IntegerMatrix categories) {
    const NumericVector& componentSamplesDims = componentSamples.attr("dim");
    unsigned int nIterations = componentSamplesDims[0];
    unsigned int nFrequencies = componentSamplesDims[1];
    unsigned int nTimes = componentSamplesDims[2];
    unsigned int nTimeSeries = categories.ncol();

    NumericVector output(Rcpp::Dimension({ nFrequencies, nTimes, nTimeSeries }));

    for (unsigned int timeSeries = 0; timeSeries < nTimeSeries; ++timeSeries) {
        for (unsigned int time = 0; time < nTimes; ++time) {
            for (unsigned int freq = 0; freq < nFrequencies; ++freq) {
                double total = 0;
                for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
                    unsigned int category = categories(iteration, timeSeries) - 1;
                    total += componentSamples[
                        category * nTimes * nFrequencies * nIterations
                        + time * nFrequencies * nIterations
                        + freq * nIterations
                        + iteration
                    ];
                }
                output[
                    timeSeries * nTimes * nFrequencies
                    + time * nFrequencies
                    + freq
                ] = total / static_cast<double>(nIterations);
            }
        }
    }

    return output;
}
