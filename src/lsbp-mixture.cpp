#include <RcppEigen.h>

#include "logging.hpp"
#include "progress.hpp"

#if defined(omp_get_num_threads)
#include "random/rng-openmp.hpp"
#endif

#include "adaptspec/samples.hpp"
#include "lsbp-mixture/sampler.hpp"
#include "lsbp-mixture/sampler-mpi.hpp"

using namespace bayesspec;

template<typename Sampler>
Rcpp::List logisticStickBreakingMixtureBase(
    unsigned int nLoop,
    unsigned int nWarmUp,
    Rcpp::NumericMatrix xR,
    Rcpp::List missingIndicesR,
    Rcpp::NumericMatrix designMatrixR,
    Rcpp::List priorsR,
    Rcpp::NumericMatrix priorMeanR,
    Rcpp::NumericMatrix priorPrecisionR,
    double tauPriorASquared, double tauPriorNu,
    double probMM1,
    double varInflate,
    double burnInVarInflate,
    bool firstCategoryFixed,
    unsigned int nSplineBases,
    Rcpp::List startR,
    Rcpp::List thin,
    bool showProgress = false
) {
    Logger logger("BayesSpec.lsbp-mixture");

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

    Eigen::MatrixXd designMatrix = Rcpp::as<Eigen::MatrixXd>(designMatrixR);
    Eigen::MatrixXd priorMean = Rcpp::as<Eigen::MatrixXd>(priorMeanR);
    Eigen::MatrixXd priorPrecision = Rcpp::as<Eigen::MatrixXd>(priorPrecisionR);
    std::vector<AdaptSpecPrior> priors = AdaptSpecPrior::fromListOfLists(priorsR);

    std::vector<AdaptSpecParameters> componentStarts = AdaptSpecParameters::fromListOfLists(
        startR["components"],
        priors
    );

    logger.debug("Constructing sampler object");
    Sampler sampler(
        x, missingIndices, designMatrix,
        probMM1, burnInVarInflate, firstCategoryFixed,
        Rcpp::as<Eigen::MatrixXd>(startR["beta"]),
        Rcpp::as<Eigen::VectorXd>(startR["tau_squared"]),
        componentStarts,
        Rcpp::as<Eigen::VectorXi>(startR["categories"]),
        priors, priorMean, priorPrecision,
        tauPriorASquared, tauPriorNu,
        nSplineBases
    );

    std::vector<AdaptSpecSamples> samples = AdaptSpecSamples::fromPriors(
        nLoop - nWarmUp,
        thin["n_segments"],
        thin["beta"],
        thin["tau_squared"],
        thin["cut_points"],
        priors
    );
    Samples<unsigned int> categoriesSamples(
        nLoop - nWarmUp,
        thin["categories"],
        designMatrix.rows(),
        true
    );
    Samples<double> betaSamples(
        nLoop - nWarmUp,
        thin["beta_lsbp"],
        {
            static_cast<std::size_t>(designMatrix.cols()),
            nComponents - 1
        }
    );
    Samples<double> tauSquaredSamples(
        nLoop - nWarmUp,
        thin["tau_squared_lsbp"],
        nComponents - 1,
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

    sampler.start();

    ProgressBar progressBar(nLoop);
    for (unsigned int iteration = 0; iteration < nLoop; ++iteration) {
        if (iteration == nWarmUp) {
            sampler.setVarInflate(varInflate);
        }

        logger.trace("[Iteration %d] Sampling", iteration);
        sampler.sample(rng);

        if (iteration % 100 == 0) {
            // NOTE(mgnb): checks whether the user has pressed Ctrl-C (among other things)
            Rcpp::checkUserInterrupt();
        }

        if (iteration >= nWarmUp) {
            logger.trace("[Iteration %d] Saving sample", iteration);
            for (unsigned int component = 0; component < nComponents; ++component) {
                samples[component].save(sampler.getParameters(component));
            }

            categoriesSamples.save(sampler.getCategories());
            betaSamples.save(sampler.getBeta());
            tauSquaredSamples.save(sampler.getTauSquared());
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
    output["beta"] = Rcpp::wrap(betaSamples);
    output["categories"] = Rcpp::wrap(categoriesSamples);
    output["tau_squared"] = Rcpp::wrap(tauSquaredSamples);
    output["log_posterior"] = Rcpp::wrap(logPosteriorSamples);
    output["x_missing"] = xMissingSamplesOutput;
    output["final_values"] = sampler.getParametersAsList();

    return output;
}

// [[Rcpp::export(name=".lsbp_mixture")]]
Rcpp::List logisticStickBreakingMixture(
    unsigned int nLoop,
    unsigned int nWarmUp,
    Rcpp::NumericMatrix xR,
    Rcpp::List missingIndicesR,
    Rcpp::NumericMatrix designMatrixR,
    Rcpp::List priorsR,
    Rcpp::NumericMatrix priorMeanR,
    Rcpp::NumericMatrix priorPrecisionR,
    double tauPriorASquared, double tauPriorNu,
    double probMM1,
    double varInflate,
    double burnInVarInflate,
    bool firstCategoryFixed,
    unsigned int nSplineBases,
    Rcpp::List startR,
    Rcpp::List thin,
    bool showProgress,
    bool mpi
) {
    if (mpi) {
        return logisticStickBreakingMixtureBase<AdaptSpecLSBPMixtureSamplerMPI>(
            nLoop,
            nWarmUp,
            xR,
            missingIndicesR,
            designMatrixR,
            priorsR,
            priorMeanR,
            priorPrecisionR,
            tauPriorASquared,
            tauPriorNu,
            probMM1,
            varInflate,
            burnInVarInflate,
            firstCategoryFixed,
            nSplineBases,
            startR,
            thin,
            showProgress
        );
    } else {
        return logisticStickBreakingMixtureBase<AdaptSpecLSBPMixtureSampler>(
            nLoop,
            nWarmUp,
            xR,
            missingIndicesR,
            designMatrixR,
            priorsR,
            priorMeanR,
            priorPrecisionR,
            tauPriorASquared,
            tauPriorNu,
            probMM1,
            varInflate,
            burnInVarInflate,
            firstCategoryFixed,
            nSplineBases,
            startR,
            thin,
            showProgress
        );
    }
}
