#include <RcppEigen.h>

using Rcpp::IntegerVector;
using Rcpp::IntegerMatrix;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

// [[Rcpp::export(name=".time_varying_mean_samples")]]
NumericMatrix timeVaryingMeanSamples(
    IntegerVector nSegments,
    IntegerMatrix cutPoints,
    NumericMatrix mu,
    unsigned int timeStep
) {
    unsigned int nIterations = nSegments.size();
    unsigned int maxTime = *std::max_element(cutPoints.begin(), cutPoints.end());
    unsigned int nTimes = std::ceil(
        static_cast<double>(maxTime) / static_cast<double>(timeStep)
    );

    NumericMatrix output(nIterations, nTimes);
    for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
        unsigned int time = 0;
        unsigned int iterationNSegments = static_cast<unsigned int>(nSegments[iteration]);
        for (unsigned int segment = 0; segment < iterationNSegments; ++segment) {
            unsigned int segmentEnd = static_cast<unsigned int>(cutPoints(iteration, segment));
            for (; time < segmentEnd; time += timeStep) {
                output(iteration, time / timeStep) = mu(iteration, segment);
            }
        }
    }

    return output;
}

// [[Rcpp::export(name=".time_varying_mean_mixture_mean_categories")]]
NumericMatrix timeVaryingMeanMixtureMeanCategories(NumericVector componentSamples, IntegerMatrix categories) {
    const NumericVector& componentSamplesDims = componentSamples.attr("dim");
    unsigned int nIterations = componentSamplesDims[0];
    unsigned int nTimes = componentSamplesDims[1];
    unsigned int nTimeSeries = categories.ncol();

    NumericMatrix output(nTimes, nTimeSeries);

    for (unsigned int timeSeries = 0; timeSeries < nTimeSeries; ++timeSeries) {
        for (unsigned int time = 0; time < nTimes; ++time) {
            double total = 0;
            for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
                unsigned int category = categories(iteration, timeSeries) - 1;
                total += componentSamples[
                    category * nTimes * nIterations
                    + time * nIterations
                    + iteration
                ];
            }
            output(time, timeSeries) = total / static_cast<double>(nIterations);
        }
    }

    return output;
}

// [[Rcpp::export(name=".time_varying_mean_mixture_mean_probabilities")]]
NumericMatrix timeVaryingMeanMixtureMeanProbabilities(NumericVector componentSamples, NumericVector probabilities) {
    const NumericVector& componentSamplesDims = componentSamples.attr("dim");
    unsigned int nIterations = componentSamplesDims[0];
    unsigned int nTimes = componentSamplesDims[1];

    const NumericVector& probabilitiesDims = probabilities.attr("dim");
    unsigned int nTimeSeries = probabilitiesDims[1];
    unsigned int nComponents = probabilitiesDims[2];

    NumericMatrix output(nTimes, nTimeSeries);

    for (unsigned int timeSeries = 0; timeSeries < nTimeSeries; ++timeSeries) {
        for (unsigned int time = 0; time < nTimes; ++time) {
            double total = 0;
            for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
                for (unsigned int component = 0; component < nComponents; ++component) {
                    total += probabilities[
                        component * nTimeSeries * nIterations
                        + timeSeries * nIterations
                        + iteration
                    ] * componentSamples[
                        component * nTimes * nIterations
                        + time * nIterations
                        + iteration
                    ];
                }

                output(time, timeSeries) = total / static_cast<double>(nIterations);
            }
        }
    }

    return output;
}
