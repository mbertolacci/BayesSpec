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
    Rcpp::List missingIndicesR,
    Rcpp::List priorsR,
    Rcpp::NumericVector weightsPriorR,
    Rcpp::List componentTuningR,
    bool firstCategoryFixed,
    Rcpp::List startR,
    Rcpp::List thin,
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

    std::vector<Eigen::VectorXi> missingIndices;
    for (Rcpp::IntegerVector missingIndicesI : missingIndicesR) {
        missingIndices.push_back(Rcpp::as<Eigen::VectorXi>(missingIndicesI));
    }
    Rcpp::List xMissingStart = startR["x_missing"];
    for (int i = 0; i < missingIndices.size(); ++i) {
        Rcpp::NumericVector xMissingStartI = xMissingStart[i];
        for (int j = 0; j < missingIndices[i].size(); ++j) {
            x(missingIndices[i][j], i) = xMissingStartI[j];
        }
    }

    std::vector<AdaptSpecPrior> priors = AdaptSpecPrior::fromListOfLists(priorsR);
    std::vector<AdaptSpecParameters> componentStarts = AdaptSpecParameters::fromListOfLists(
        startR["components"],
        priors
    );
    AdaptSpecTuning componentTuning = AdaptSpecTuning::fromList(componentTuningR);

    AdaptSpecIndependentMixtureSampler sampler(
        x, missingIndices,
        componentTuning, firstCategoryFixed,
        Rcpp::as<Eigen::VectorXd>(startR["weights"]),
        componentStarts,
        Rcpp::as<Eigen::VectorXi>(startR["categories"]),
        priors,
        Rcpp::as<Eigen::VectorXd>(weightsPriorR)
    );

    std::vector<AdaptSpecSamples> samples = AdaptSpecSamples::fromPriors(
        nLoop - nWarmUp,
        thin["n_segments"],
        thin["beta"],
        thin["tau_squared"],
        thin["cut_points"],
        thin["mu"],
        priors
    );
    Samples<unsigned int> categoriesSamples(
        nLoop - nWarmUp,
        thin["categories"],
        x.cols(),
        true
    );
    Samples<double> weightsSamples(
        nLoop - nWarmUp,
        thin["weights"],
        nComponents,
        true
    );
    Samples<double> logPosteriorSamples(nLoop - nWarmUp, thin["log_posterior"]);
    std::vector< Samples<double> > xMissingSamples;
    for (int i = 0; i < missingIndices.size(); ++i) {
        xMissingSamples.emplace_back(
            nLoop - nWarmUp,
            thin["x_missing"],
            missingIndices[i].size()
        );
    }

    ProgressBar progressBar(nLoop);
    for (unsigned int iteration = 0; iteration < nLoop; ++iteration) {
        if (iteration == nWarmUp) {
            sampler.endWarmUp();
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

            categoriesSamples.save(sampler.getCategories());
            weightsSamples.save(sampler.getWeights());
            logPosteriorSamples.save(sampler.getLogPosterior());
            for (int i = 0; i < missingIndices.size(); ++i) {
                if (missingIndices[i].size() == 0) continue;
                std::vector<double> xMissing(missingIndices[i].size());
                for (int j = 0; j < missingIndices[i].size(); ++j) {
                    xMissing[j] = x(missingIndices[i][j], i);
                }
                xMissingSamples[i].save(xMissing);
            }
        }

        if (showProgress) {
            ++progressBar;
        }
    }

    Rcpp::List components;
    for (unsigned int component = 0; component < nComponents; ++component) {
        components.push_back(samples[component].asList());
    }
    Rcpp::List xMissingSamplesOutput;
    for (Samples<double> samples : xMissingSamples) {
        xMissingSamplesOutput.push_back(Rcpp::wrap(samples));
    }
    Rcpp::List output;
    output["components"] = components;
    output["weights"] = Rcpp::wrap(weightsSamples);
    output["categories"] = Rcpp::wrap(categoriesSamples);
    output["log_posterior"] = Rcpp::wrap(logPosteriorSamples);
    output["x_missing"] = xMissingSamplesOutput;
    output["final_values"] = sampler.getParametersAsList();
    output["component_statistics"] = sampler.getComponentStatistics();
    output["component_warm_up_statistics"] = sampler.getComponentWarmUpStatistics();

    return output;
}
