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
    unsigned int nFrequencies,
    unsigned int timeStep,
    std::string frequencyTransform
) {
    const NumericVector& betaDims = beta.attr("dim");

    unsigned int nIterations = nSegments.size();
    unsigned int nSegmentsMax = cutPoints.ncol();
    unsigned int maxTime = *std::max_element(cutPoints.begin(), cutPoints.end());
    unsigned int nTimes = std::ceil(
        static_cast<double>(maxTime) / static_cast<double>(timeStep)
    );
    unsigned int nBeta = betaDims[2];
    unsigned int nBases = nBeta - 1;

    VectorXd frequencies = VectorXd::LinSpaced(
        nFrequencies, 0, nFrequencies - 1
    ) / static_cast<double>(2 * (nFrequencies - 1));
    VectorXd transformedFrequencies(nFrequencies);
    if (frequencyTransform == "cbrt") {
        for (int i = 0; i < frequencies.size(); ++i) {
            transformedFrequencies[i] = std::cbrt(2 * frequencies[i]);
        }
    } else {
        transformedFrequencies = 2 * frequencies;
    }
    Eigen::MatrixXd nuHat = splineBasis1dDemmlerReinsch(transformedFrequencies, nBases);
    VectorXd segmentBeta(nBeta);

    NumericVector output(Rcpp::Dimension({ nIterations, nFrequencies, nTimes }));
    for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
        unsigned int time = 0;
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

            unsigned int segmentEnd = static_cast<unsigned int>(cutPoints(iteration, segment));
            for (; time < segmentEnd; time += timeStep) {
                for (unsigned int freq = 0; freq < nFrequencies; ++freq) {
                    output[
                        (time / timeStep) * nFrequencies * nIterations
                        + freq * nIterations
                        + iteration
                    ] = segmentSpectra[freq];
                }
            }
        }
    }
    output.attr("frequencies") = Rcpp::wrap(frequencies);

    return output;
}

// [[Rcpp::export(name=".time_varying_spectra_mixture_mean_categories")]]
NumericVector timeVaryingSpectraMixtureMeanCategories(NumericVector componentSamples, IntegerMatrix categories) {
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

// [[Rcpp::export(name=".time_varying_spectra_mixture_mean_probabilities")]]
NumericVector timeVaryingSpectraMixtureMeanProbabilities(NumericVector componentSamples, NumericVector probabilities) {
    const NumericVector& componentSamplesDims = componentSamples.attr("dim");
    unsigned int nIterations = componentSamplesDims[0];
    unsigned int nFrequencies = componentSamplesDims[1];
    unsigned int nTimes = componentSamplesDims[2];

    const NumericVector& probabilitiesDims = probabilities.attr("dim");
    unsigned int nTimeSeries = probabilitiesDims[1];
    unsigned int nComponents = probabilitiesDims[2];

    NumericVector output(Rcpp::Dimension({ nFrequencies, nTimes, nTimeSeries }));

    for (unsigned int timeSeries = 0; timeSeries < nTimeSeries; ++timeSeries) {
        for (unsigned int time = 0; time < nTimes; ++time) {
            for (unsigned int freq = 0; freq < nFrequencies; ++freq) {
                double total = 0;
                for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
                    for (unsigned int component = 0; component < nComponents; ++component) {
                        total += probabilities[
                            component * nTimeSeries * nIterations
                            + timeSeries * nIterations
                            + iteration
                        ] * componentSamples[
                            component * nTimes * nFrequencies * nIterations
                            + time * nFrequencies * nIterations
                            + freq * nIterations
                            + iteration
                        ];
                    }
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
