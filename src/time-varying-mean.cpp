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
    IntegerVector times
) {
    unsigned int nIterations = nSegments.size();
    unsigned int nTimes = times.size();

    NumericMatrix output(nIterations, nTimes);
    for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
        unsigned int timeIndex = 0;
        unsigned int iterationNSegments = static_cast<unsigned int>(nSegments[iteration]);
        for (unsigned int segment = 0; segment < iterationNSegments; ++segment) {
            unsigned int segmentEnd = static_cast<unsigned int>(cutPoints(iteration, segment));
            for (; timeIndex < nTimes && times[timeIndex] <= segmentEnd; ++timeIndex) {
                output(iteration, timeIndex) = mu(iteration, segment);
            }
        }
    }
    output.attr("times") = times;

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

    for (unsigned int component = 0; component < nComponents; ++component) {
        for (unsigned int timeSeries = 0; timeSeries < nTimeSeries; ++timeSeries) {
            for (unsigned int time = 0; time < nTimes; ++time) {
                double total = 0;
                for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
                    total += probabilities[
                        component * nTimeSeries * nIterations
                        + timeSeries * nIterations
                        + iteration
                    ] * componentSamples[
                        component * nTimes * nIterations
                        + time * nIterations
                        + iteration
                    ] / static_cast<double>(nIterations);
                }
                output(time, timeSeries) += total;
            }
        }
    }

    return output;
}

// [[Rcpp::export(name=".time_varying_mean_mixture_samples_probabilities")]]
NumericVector timeVaryingMeanMixtureSamplesProbabilities(NumericVector componentSamples, NumericVector probabilities) {
    const NumericVector& componentSamplesDims = componentSamples.attr("dim");
    unsigned int nIterations = componentSamplesDims[0];
    unsigned int nTimes = componentSamplesDims[1];

    const NumericVector& probabilitiesDims = probabilities.attr("dim");
    unsigned int nTimeSeries = probabilitiesDims[1];
    unsigned int nComponents = probabilitiesDims[2];

    NumericVector output(Rcpp::Dimension(nIterations, nTimes, nTimeSeries));

    for (unsigned int timeSeries = 0; timeSeries < nTimeSeries; ++timeSeries) {
        for (unsigned int time = 0; time < nTimes; ++time) {
            for (unsigned int iteration = 0; iteration < nIterations; ++iteration) {
                double total = 0;
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

                output[
                    timeSeries * nTimes * nIterations
                    + time * nIterations
                    + iteration
                ] = total;
            }
        }
    }

    return output;
}
