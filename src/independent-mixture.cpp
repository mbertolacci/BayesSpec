#include <RcppEigen.h>

#include "progress.hpp"

#include "adaptspec/samples.hpp"
#include "independent-mixture/sampler.hpp"

using namespace bayesspec;

// [[Rcpp::export(name=".independent_mixture")]]
Rcpp::List independentMixture(
    unsigned int nLoop,
    unsigned int nWarmUp,
    Rcpp::NumericMatrix xR,
    Rcpp::List priorsR,
    Rcpp::NumericVector weightsPriorR,
    double probMM1,
    bool showProgress = false
) {
    RNG::initialise();
    Eigen::initParallel();

    unsigned int nComponents = priorsR.size();
    Eigen::MatrixXd x = Rcpp::as<Eigen::MatrixXd>(xR);

    std::vector<AdaptSpecParameters> starts;
    std::vector<AdaptSpecPrior> priors;
    // We reserve space in these so that pointers into them will last
    starts.reserve(nComponents);
    priors.reserve(nComponents);
    std::vector<AdaptSpecSamples> samples;
    for (unsigned int component = 0; component < nComponents; ++component) {
        priors.push_back(AdaptSpecPrior::fromList(priorsR[component]));
        starts.emplace_back(priors[component], x.rows(), 1);
        samples.emplace_back(nLoop - nWarmUp, priors[component]);
    }

    AdaptSpecIndependentMixtureSampler sampler(
        x, probMM1, starts, priors, Rcpp::as<Eigen::VectorXd>(weightsPriorR)
    );

    Rcpp::IntegerMatrix categoriesSamples(x.cols(), nLoop - nWarmUp);
    Rcpp::NumericMatrix weightsSamples(nComponents, nLoop - nWarmUp);

    ProgressBar progressBar(nLoop);
    for (unsigned int iteration = 0; iteration < nLoop; ++iteration) {
        sampler.sample();

        if (iteration % 100 == 0) {
            // NOTE(mgnb): checks whether the user has pressed Ctrl-C (among other things)
            Rcpp::checkUserInterrupt();
        }

        if (iteration >= nWarmUp) {
            for (unsigned int component = 0; component < nComponents; ++component) {
                samples[component].save(sampler.getParameters(component));
            }

            unsigned int sampleIndex = iteration - nWarmUp;
            std::copy(
                sampler.getCategories().data(),
                sampler.getCategories().data() + x.cols(),
                categoriesSamples.begin() + sampleIndex * x.cols()
            );
            std::copy(
                sampler.getWeights().data(),
                sampler.getWeights().data() + nComponents,
                weightsSamples.begin() + sampleIndex * nComponents
            );
        }

        if (showProgress) {
            ++progressBar;
        }
    }

    Rcpp::List results;
    Rcpp::List components;
    for (unsigned int component = 0; component < nComponents; ++component) {
        components.push_back(samples[component].asList());
    }
    results["components"] = components;
    results["weights"] = weightsSamples;
    results["categories"] = categoriesSamples;

    return results;
}
