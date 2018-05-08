#include <RcppEigen.h>

#include "progress.hpp"

#if defined(omp_get_num_threads)
#include "random/rng-openmp.hpp"
#endif

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
    Rcpp::IntegerVector initialCategoriesR,
    double probMM1,
    double varInflate,
    double burnInVarInflate,
    bool firstCategoryFixed,
    bool showProgress = false
) {
    #if defined(omp_get_num_threads)
        Eigen::initParallel();
        RNGOpenMP<std::mt19937_64> rng([]() -> uint_fast64_t {
            return static_cast<uint_fast64_t>(UINT_FAST64_MAX * R::unif_rand());
        });
    #else
        std::mt19937_64 rng(static_cast<uint_fast64_t>(UINT_FAST64_MAX * R::unif_rand()));
    #endif

    unsigned int nComponents = priorsR.size();
    Eigen::MatrixXd x = Rcpp::as<Eigen::MatrixXd>(xR);

    std::vector<AdaptSpecPrior> priors = AdaptSpecPrior::fromListOfLists(priorsR);
    std::vector<AdaptSpecSamples> samples = AdaptSpecSamples::fromPriors(
        nLoop - nWarmUp,
        priors
    );
    Eigen::VectorXd weightsPrior = Rcpp::as<Eigen::VectorXd>(weightsPriorR);
    Eigen::VectorXi initialCategories = Rcpp::as<Eigen::VectorXi>(initialCategoriesR);

    AdaptSpecIndependentMixtureSampler sampler(
        x, probMM1, burnInVarInflate, firstCategoryFixed,
        // starts,
        initialCategories,
        priors,
        weightsPrior
    );

    Rcpp::IntegerMatrix categoriesSamples(x.cols(), nLoop - nWarmUp);
    Rcpp::NumericMatrix weightsSamples(nComponents, nLoop - nWarmUp);
    Rcpp::NumericVector logPosteriorSamples(nLoop - nWarmUp);

    ProgressBar progressBar(nLoop);
    for (unsigned int iteration = 0; iteration < nLoop; ++iteration) {
        if (iteration == nWarmUp) {
            sampler.setVarInflate(varInflate);
        }

        sampler.sample(rng);

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
            logPosteriorSamples[sampleIndex] = sampler.getLogPosterior();
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
    results["log_posterior"] = logPosteriorSamples;

    return results;
}
