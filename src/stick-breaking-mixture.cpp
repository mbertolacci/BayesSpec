#include <RcppEigen.h>

#include "progress.hpp"

#if defined(omp_get_num_threads)
#include "random/rng-openmp.hpp"
#endif

#include "adaptspec/samples.hpp"
#include "stick-breaking-mixture/sampler.hpp"

using namespace bayesspec;

// [[Rcpp::export(name=".stick_breaking_mixture")]]
Rcpp::List stickBreakingMixture(
    unsigned int nLoop,
    unsigned int nWarmUp,
    Rcpp::NumericMatrix xR,
    Rcpp::NumericMatrix designMatrixR,
    Rcpp::List priorsR,
    Rcpp::NumericMatrix priorMeanR,
    Rcpp::NumericMatrix priorPrecisionR,
    double tauPriorASquared, double tauPriorNu,
    Rcpp::IntegerVector initialCategoriesR,
    double probMM1,
    double varInflate,
    double burnInVarInflate,
    bool firstCategoryFixed,
    unsigned int nSplineBases,
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
    Eigen::MatrixXd designMatrix = Rcpp::as<Eigen::MatrixXd>(designMatrixR);
    Eigen::MatrixXd priorMean = Rcpp::as<Eigen::MatrixXd>(priorMeanR);
    Eigen::MatrixXd priorPrecision = Rcpp::as<Eigen::MatrixXd>(priorPrecisionR);

    std::vector<AdaptSpecParameters> starts;
    std::vector<AdaptSpecPrior> priors;
    // We reserve space in these so that pointers into them will last
    starts.reserve(nComponents);
    priors.reserve(nComponents);
    std::vector<AdaptSpecSamples> samples;
    for (unsigned int component = 0; component < nComponents; ++component) {
        priors.push_back(AdaptSpecPrior::fromList(priorsR[component]));
        starts.emplace_back(priors[component], x.rows());
        samples.emplace_back(nLoop - nWarmUp, priors[component]);
    }
    Eigen::VectorXi initialCategories = Rcpp::as<Eigen::VectorXi>(initialCategoriesR);

    AdaptSpecStickBreakingMixtureSampler sampler(
        x, designMatrix,
        probMM1, burnInVarInflate, firstCategoryFixed,
        starts, initialCategories,
        priors, priorMean, priorPrecision,
        tauPriorASquared, tauPriorNu,
        nSplineBases
    );

    Rcpp::IntegerMatrix categoriesSamples(x.cols(), nLoop - nWarmUp);
    unsigned int nBetas = priorMean.rows() * priorMean.cols();
    Rcpp::NumericVector betaSamples(Rcpp::Dimension({
        designMatrix.cols(),
        nComponents - 1,
        nLoop - nWarmUp
    }));
    Rcpp::NumericMatrix tauSquaredSamples(nComponents - 1, nLoop - nWarmUp);
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
                sampler.getBeta().data(),
                sampler.getBeta().data() + nBetas,
                betaSamples.begin() + sampleIndex * nBetas
            );
            std::copy(
                sampler.getTauSquared().data(),
                sampler.getTauSquared().data() + nComponents - 1,
                tauSquaredSamples.begin() + sampleIndex * (nComponents - 1)
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
    results["beta"] = betaSamples;
    results["categories"] = categoriesSamples;
    results["tau_squared"] = tauSquaredSamples;
    results["log_posterior"] = logPosteriorSamples;

    return results;
}
