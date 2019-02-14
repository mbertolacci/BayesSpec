#ifndef SRC_LSBP_SAMPLER_HPP_
#define SRC_LSBP_SAMPLER_HPP_

#include <RcppEigen.h>
#include <algorithm>

#include "../mixture-base/sampler-base.hpp"
#include "sampler-strategy.hpp"
#include "independent-proposal.hpp"

namespace bayesspec {

// log(1 + exp(x))
double log1pexp(double x) {
    // At this point, the difference between log(1 + exp(x)) and x is below
    // machine precision, and this prevents overflow
    if (x > 36.04365) return x;
    return std::log1p(std::exp(x));
}

template<
    typename AdaptSpecLSBPMixtureStrategyType,
    typename MixtureBaseStrategyType
>
class AdaptSpecLSBPMixtureSamplerBase
    : public AdaptSpecLSBPMixtureStrategyType,
      public MixtureSamplerBase<
        AdaptSpecLSBPMixtureSamplerBase<
            AdaptSpecLSBPMixtureStrategyType,
            MixtureBaseStrategyType
        >,
        MixtureBaseStrategyType
    >
{
public:
    typedef MixtureSamplerBase<
        AdaptSpecLSBPMixtureSamplerBase<
            AdaptSpecLSBPMixtureStrategyType,
            MixtureBaseStrategyType
        >,
        MixtureBaseStrategyType
    > Base;

    AdaptSpecLSBPMixtureSamplerBase(
        Eigen::MatrixXd& x,
        const std::vector<Eigen::VectorXi>& missingIndices,
        const Eigen::MatrixXd& designMatrix,
        const AdaptSpecTuning& componentTuning,
        bool firstCategoryFixed,
        const Eigen::MatrixXd& betaStart,
        const Eigen::VectorXd& tauSquaredStart,
        const std::vector<AdaptSpecParameters>& componentStart,
        const Eigen::VectorXi& categoriesStart,
        const std::vector<AdaptSpecPrior>& componentPriors,
        const Eigen::MatrixXd& priorMean,
        const Eigen::MatrixXd& priorPrecision,
        double tauPriorASquared, double tauPriorNu, double tauPriorUpper,
        unsigned int nSplineBases,
        unsigned int nSwapMoves,
        unsigned int swapMoveLength,
        unsigned int nSplitMergeMoves
    ) : Base(
            x, missingIndices,
            componentTuning, firstCategoryFixed,
            componentStart, categoriesStart,
            componentPriors
        ),
        designMatrix_(designMatrix),
        priorMean_(priorMean),
        priorPrecision_(priorPrecision),
        tauPriorASquared_(tauPriorASquared),
        tauPriorNu_(tauPriorNu),
        tauPriorUpper_(tauPriorUpper),
        parameters_(betaStart),
        tauSquared_(tauSquaredStart),
        nSplineBases_(nSplineBases),
        nSwapMoves_(nSwapMoves),
        swapMoveLength_(swapMoveLength),
        nSplitMergeMoves_(nSplitMergeMoves),
        nSwaps_(0),
        acceptedSwaps_(0),
        nMerges_(0),
        acceptedMerges_(0),
        nSplits_(0),
        acceptedSplits_(0) {
        updateWeights_();
    }

    template<typename RNG>
    void sampleWeights_(RNG& rng) {
        this->sampleLSBPWeights_(
            parameters_,
            tauSquared_,
            designMatrix_,
            priorMean_,
            priorPrecision_,
            this->categories_,
            this->counts_,
            tauPriorNu_,
            tauPriorASquared_,
            tauPriorUpper_,
            nSplineBases_,
            rng
        );

        if (nSplineBases_ > 0) {
            updatePriorPrecision_();
        }
        updateWeights_();

        for (unsigned int i = 0; i < nSwapMoves_; ++i) {
            sampleSwap_(rng);
        }
        for (unsigned int i = 0; i < nSplitMergeMoves_; ++i) {
            sampleSplitMerge_(rng);
        }
    }

    const Eigen::MatrixXd& getBeta() const {
        return parameters_;
    }

    const Eigen::VectorXd& getTauSquared() const {
        return tauSquared_;
    }

    double getWeightsLogPrior_() const {
        double output = (
            priorPrecision_.array().log().sum() / 2.0
            - ((parameters_ - priorMean_).array().square() * priorPrecision_.array()).sum() / 2.0
        );
        if (nSplineBases_ > 0) {
            output -= (tauPriorNu_ + 1) / 2.0 * (1.0 + tauSquared_.array() / (tauPriorNu_ * tauPriorASquared_)).log().sum();
        }
        return output;
    }

    Rcpp::List getWeightsParametersAsList() const {
        Rcpp::List output;
        output["beta"] = Rcpp::wrap(parameters_);
        output["tau_squared"] = Rcpp::wrap(tauSquared_);
        return output;
    }

    Rcpp::List getStatisticsAsList() const {
        Rcpp::List output;
        output["n_swaps"] = nSwaps_;
        output["accepted_swaps"] = acceptedSwaps_;
        output["n_merges"] = nMerges_;
        output["accepted_merges"] = acceptedMerges_;
        output["n_splits"] = nSplits_;
        output["accepted_splits"] = acceptedSplits_;
        return output;
    }

private:
    Eigen::MatrixXd designMatrix_;

    Eigen::MatrixXd priorMean_;
    Eigen::MatrixXd priorPrecision_;
    double tauPriorASquared_;
    double tauPriorNu_;
    double tauPriorUpper_;

    Eigen::MatrixXd parameters_;
    Eigen::VectorXd tauSquared_;

    unsigned int nSplineBases_;

    unsigned int nSwapMoves_;
    unsigned int swapMoveLength_;
    unsigned int nSplitMergeMoves_;

    unsigned int nSwaps_;
    unsigned int acceptedSwaps_;
    unsigned int nMerges_;
    unsigned int acceptedMerges_;
    unsigned int nSplits_;
    unsigned int acceptedSplits_;

    void updatePriorPrecision_() {
        unsigned int splineStartIndex = parameters_.rows() - nSplineBases_;
        for (unsigned int component = 0; component < this->nComponents_ - 1; ++component) {
            for (unsigned int k = 0; k < nSplineBases_; ++k) {
                priorPrecision_(splineStartIndex + k, component) = 1.0 / tauSquared_[component];
            }
        }
    }

    void updateWeights_() {
        Eigen::MatrixXd values = (designMatrix_ * parameters_).array().matrix();

        for (unsigned int series = 0; series < designMatrix_.rows(); ++series) {
            double sumAccumulator = 0;
            for (unsigned int component = 0; component < this->nComponents_ - 1; ++component) {
                this->allLogWeights_(series, component) = (
                    -log1pexp(-values(series, component)) + sumAccumulator
                );
                sumAccumulator = sumAccumulator - log1pexp(values(series, component));
            }
            this->allLogWeights_(series, this->nComponents_ - 1) = sumAccumulator;
        }
    }

    template<typename RNG>
    void sampleSwap_(RNG& rng) {
        unsigned int lower = randInteger(0, this->nComponents_ - 2, rng);
        unsigned int upper = randInteger(
            lower + 1,
            std::min(lower + swapMoveLength_, this->nComponents_ - 1),
            rng
        );

        Eigen::VectorXi oldCategories(this->categories_);
        Eigen::MatrixXd oldParameters(parameters_);
        Eigen::VectorXd oldTauSquared(tauSquared_);
        Eigen::MatrixXd oldAllLogWeights(this->allLogWeights_);
        double oldLogPosterior = this->getLogPosterior();

        LSBPIndependentProposal lowerProposalFrom(
            lower,
            this->categories_,
            priorMean_.col(lower),
            priorPrecision_.col(lower),
            designMatrix_,
            parameters_.col(lower)
        );
        double fromLogProposal = lowerProposalFrom.logDensity(parameters_.col(lower));

        if (upper != this->nComponents_ - 1) {
            LSBPIndependentProposal upperProposalFrom(
                upper,
                this->categories_,
                priorMean_.col(upper),
                priorPrecision_.col(upper),
                designMatrix_,
                parameters_.col(upper)
            );
            fromLogProposal += upperProposalFrom.logDensity(parameters_.col(upper));

            std::swap(tauSquared_[lower], tauSquared_[upper]);
            updatePriorPrecision_();
        }
        for (unsigned int i = 0; i < this->categories_.size(); ++i) {
            if (this->categories_[i] == lower) {
                this->categories_[i] = upper;
            } else if (this->categories_[i] == upper) {
                this->categories_[i] = lower;
            }
        }

        LSBPIndependentProposal lowerProposalTo(
            lower,
            this->categories_,
            priorMean_.col(lower),
            priorPrecision_.col(lower),
            designMatrix_,
            randNormal(parameters_.rows(), rng)
        );
        parameters_.col(lower) = lowerProposalTo.sample(rng);
        double toLogProposal = lowerProposalTo.logDensity(parameters_.col(lower));
        if (upper != this->nComponents_ - 1) {
            LSBPIndependentProposal upperProposalTo(
                upper,
                this->categories_,
                priorMean_.col(upper),
                priorPrecision_.col(upper),
                designMatrix_,
                randNormal(parameters_.rows(), rng)
            );
            parameters_.col(upper) = upperProposalTo.sample(rng);
            toLogProposal += upperProposalTo.logDensity(parameters_.col(upper));
        }

        updateWeights_();

        double newLogPosterior = this->getLogPosterior();
        double logAccept = (
            newLogPosterior - oldLogPosterior
            + fromLogProposal - toLogProposal
        );
        if (this->isWarmedUp()) ++nSwaps_;
        if (randUniform(rng) < std::exp(logAccept)) {
            if (this->isWarmedUp()) ++acceptedSwaps_;
            std::swap(this->componentStates_[lower], this->componentStates_[upper]);
            this->updateCounts_();
        } else {
            this->categories_ = oldCategories;
            parameters_ = oldParameters;
            tauSquared_ = oldTauSquared;
            this->allLogWeights_ = oldAllLogWeights;
            updatePriorPrecision_();
        }
    }

    template<typename RNG>
    void sampleSplitMerge_(RNG& rng) {
        std::vector<unsigned int> indices;
        for (unsigned int k = 0; k < this->nComponents_; ++k) {
            indices.push_back(k);
        }
        std::shuffle(indices.begin(), indices.end(), rng);
        unsigned int componentA = indices[0];
        unsigned int componentB = indices[1];

        unsigned int rightMostParameters = this->nComponents_ - 2;

        Eigen::VectorXi oldCategories(this->categories_);
        Eigen::MatrixXd oldParameters(parameters_);
        Eigen::VectorXi oldCounts(this->counts_);
        Eigen::VectorXd oldTauSquared(tauSquared_);
        Eigen::MatrixXd oldPriorPrecision(priorPrecision_);
        Eigen::MatrixXd oldAllLogWeights(this->allLogWeights_);
        std::vector<AdaptSpecMixtureComponentState> oldComponentStates(this->componentStates_);
        double oldLogPosterior = this->getLogPosterior();

        if (randUniform(rng) < 0.5) {
            if (this->isWarmedUp()) ++nMerges_;
            if (this->counts_[componentA] == 0 && this->counts_[componentB] == 0) return;

            // Merge b => a. Do this by reassigning members of b to
            // a, then renumbering all components >= b down one
            double parameterFromLogProposal = 0;
            if (componentA < this->nComponents_ - 1) {
                LSBPIndependentProposal aProposalFrom(componentA, oldCategories, priorMean_.col(componentA), oldPriorPrecision.col(componentA), designMatrix_, randNormal(parameters_.rows(), rng));
                parameterFromLogProposal += aProposalFrom.logDensity(oldParameters.col(componentA));
            }
            if (componentB < this->nComponents_ - 1) {
                LSBPIndependentProposal bProposalFrom(componentB, oldCategories, priorMean_.col(componentB), oldPriorPrecision.col(componentB), designMatrix_, randNormal(parameters_.rows(), rng));
                parameterFromLogProposal += bProposalFrom.logDensity(oldParameters.col(componentB));
            }

            unsigned int shiftedComponentA = componentA;
            if (componentA > componentB) {
                shiftedComponentA = componentA - 1;
            }
            for (unsigned int i = 0; i < this->categories_.size(); ++i) {
                if (this->categories_[i] == componentB) {
                    this->categories_[i] = shiftedComponentA;
                } else if (this->categories_[i] > componentB) {
                    this->categories_[i]--;
                }
            }
            this->updateCounts_();

            // Shift everything down a slot
            for (unsigned int k = componentB; k < this->nComponents_ - 2; ++k) {
                parameters_.col(k) = parameters_.col(k + 1);
                tauSquared_[k] = tauSquared_[k + 1];
                this->componentStates_[k] = this->componentStates_[k + 1];
            }
            if (componentB < this->nComponents_ - 1) {
                this->componentStates_[this->nComponents_ - 2] = this->componentStates_[this->nComponents_ - 1];
            }
            // The rightmost set of parameters need some values
            this->componentStates_[this->nComponents_ - 1] = oldComponentStates[componentB];
            if (componentB < this->nComponents_ - 1) {
                parameters_.col(rightMostParameters) = oldParameters.col(componentB);
                tauSquared_[rightMostParameters] = oldTauSquared[componentB];
            }
            updatePriorPrecision_();

            LSBPIndependentProposal aProposalTo(shiftedComponentA, this->categories_, priorMean_.col(shiftedComponentA), priorPrecision_.col(shiftedComponentA), designMatrix_, randNormal(parameters_.rows(), rng));
            parameters_.col(shiftedComponentA) = aProposalTo.sample(rng);
            double parameterToLogProposal = aProposalTo.logDensity(parameters_.col(shiftedComponentA));
            if (componentB < this->nComponents_ - 1) {
                LSBPIndependentProposal rightMostProposalTo(rightMostParameters, this->categories_, priorMean_.col(rightMostParameters), priorPrecision_.col(rightMostParameters), designMatrix_, randNormal(parameters_.rows(), rng));
                parameters_.col(rightMostParameters) = rightMostProposalTo.sample(rng);
                parameterToLogProposal += rightMostProposalTo.logDensity(parameters_.col(rightMostParameters));
            }
            updateWeights_();

            this->componentStates_[shiftedComponentA].proposeSpectra(
                this->categories_.array() == static_cast<int>(shiftedComponentA),
                this->counts_[shiftedComponentA],
                rng
            );
            this->componentStates_[this->nComponents_ - 1].proposeSpectra(
                this->categories_.array() == static_cast<int>(this->nComponents_ - 1),
                this->counts_[this->nComponents_ - 1],
                rng
            );

            double logAccept = (
                this->getLogPosterior() - oldLogPosterior
                + std::log(0.5)
                - std::log(0.5)
                + this->counts_[shiftedComponentA] * std::log(0.5)
                - 0
                + parameterFromLogProposal
                - parameterToLogProposal
                + oldComponentStates[componentA].getLogSegmentProposal()
                + oldComponentStates[componentB].getLogSegmentProposal()
                - this->componentStates_[shiftedComponentA].getLogSegmentProposal()
                - this->componentStates_[this->nComponents_ - 1].getLogSegmentProposal()
            );
            if (randUniform(rng) < std::exp(logAccept)) {
                if (this->isWarmedUp()) ++acceptedMerges_;
            } else {
                this->categories_ = oldCategories;
                parameters_ = oldParameters;
                this->updateCounts_();
                tauSquared_ = oldTauSquared;
                priorPrecision_ = oldPriorPrecision;
                this->allLogWeights_ = oldAllLogWeights;
                this->componentStates_ = oldComponentStates;
            }
        } else {
            if (this->isWarmedUp()) ++nSplits_;
            // HACK(mgnb): need to do the maths to take this into account
            if (this->counts_[this->nComponents_ - 1] > 0) return;

            // Split a => a, b. Do this by renumbering all components >= b,
            // then randomly assigning members of a either a or b
            unsigned int shiftedComponentA = componentA;
            if (componentA > componentB) {
                shiftedComponentA = componentA - 1;
            }

            if (this->counts_[shiftedComponentA] == 0) return;

            double parameterFromLogProposal = 0;
            LSBPIndependentProposal aProposalFrom(shiftedComponentA, oldCategories, priorMean_.col(shiftedComponentA), oldPriorPrecision.col(shiftedComponentA), designMatrix_, randNormal(parameters_.rows(), rng));
            parameterFromLogProposal += aProposalFrom.logDensity(oldParameters.col(shiftedComponentA));
            if (componentB < this->nComponents_ - 1) {
                LSBPIndependentProposal bProposalFrom(this->nComponents_ - 2, oldCategories, priorMean_.col(this->nComponents_ - 2), oldPriorPrecision.col(this->nComponents_ - 2), designMatrix_, randNormal(parameters_.rows(), rng));
                parameterFromLogProposal += bProposalFrom.logDensity(oldParameters.col(this->nComponents_ - 2));
            }

            for (unsigned int i = 0; i < this->categories_.size(); ++i) {
                if (this->categories_[i] == shiftedComponentA) {
                    this->categories_[i] = randUniform(rng) < 0.5 ? componentA : componentB;
                } else if (this->categories_[i] >= componentB) {
                    this->categories_[i]++;
                }
            }
            this->updateCounts_();

            // Shift everything up a slot
            if (componentB < this->nComponents_ - 1) {
                this->componentStates_[this->nComponents_ - 1] = this->componentStates_[this->nComponents_ - 2];
            }
            for (unsigned int k = this->nComponents_ - 2; k > componentB; --k) {
                parameters_.col(k) = parameters_.col(k - 1);
                tauSquared_[k] = tauSquared_[k - 1];
                this->componentStates_[k] = this->componentStates_[k - 1];
            }
            // componentB needs some values
            this->componentStates_[componentB] = oldComponentStates[this->nComponents_ - 1];
            if (componentB < this->nComponents_ - 1) {
                parameters_.col(componentB) = oldParameters.col(this->nComponents_ - 2);
                tauSquared_[componentB] = oldTauSquared[this->nComponents_ - 2];
            }
            updatePriorPrecision_();

            double parameterToLogProposal = 0;
            if (componentA < this->nComponents_ - 1) {
                LSBPIndependentProposal aProposalTo(componentA, this->categories_, priorMean_.col(componentA), priorPrecision_.col(componentA), designMatrix_, randNormal(parameters_.rows(), rng));
                parameters_.col(componentA) = aProposalTo.sample(rng);
                parameterToLogProposal += aProposalTo.logDensity(parameters_.col(componentA));
            }
            if (componentB < this->nComponents_ - 1) {
                LSBPIndependentProposal bProposalTo(componentB, this->categories_, priorMean_.col(componentB), priorPrecision_.col(componentB), designMatrix_, randNormal(parameters_.rows(), rng));
                parameters_.col(componentB) = bProposalTo.sample(rng);
                parameterToLogProposal += bProposalTo.logDensity(parameters_.col(componentB));
            }
            updateWeights_();

            this->componentStates_[componentA].proposeSpectra(
                this->categories_.array() == static_cast<int>(componentA),
                this->counts_[componentA],
                rng
            );
            this->componentStates_[componentB].proposeSpectra(
                this->categories_.array() == static_cast<int>(componentB),
                this->counts_[componentB],
                rng
            );

            double logAccept = (
                this->getLogPosterior() - oldLogPosterior
                + std::log(0.5)
                - std::log(0.5)
                + 0
                - (this->counts_[componentA] + this->counts_[componentB]) * std::log(0.5)
                + parameterFromLogProposal
                - parameterToLogProposal
                + oldComponentStates[shiftedComponentA].getLogSegmentProposal()
                + oldComponentStates[this->nComponents_ - 1].getLogSegmentProposal()
                - this->componentStates_[componentA].getLogSegmentProposal()
                - this->componentStates_[componentB].getLogSegmentProposal()
            );
            if (randUniform(rng) < std::exp(logAccept)) {
                if (this->isWarmedUp()) ++acceptedSplits_;
            } else {
                this->categories_ = oldCategories;
                parameters_ = oldParameters;
                this->updateCounts_();
                tauSquared_ = oldTauSquared;
                priorPrecision_ = oldPriorPrecision;
                this->allLogWeights_ = oldAllLogWeights;
                this->componentStates_ = oldComponentStates;
            }
        }
    }
};

typedef AdaptSpecLSBPMixtureSamplerBase<
    AdaptSpecLSBPMixtureStrategy,
    MixtureBaseStrategy
> AdaptSpecLSBPMixtureSampler;

}  // namespace bayespec

#endif  // SRC_LSBP_SAMPLER_HPP_
