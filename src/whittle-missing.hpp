#ifndef SRC_WHITTLE_MISSING_HPP_
#define SRC_WHITTLE_MISSING_HPP_

#include <RcppEigen.h>

#include "random/utils.hpp"
#include "fft.hpp"

namespace bayesspec {

class WhittleMissingValuesDistribution {
public:
    WhittleMissingValuesDistribution()
        : missingIndices_(0),
          missingMean_(0) {}

    WhittleMissingValuesDistribution(
        const Eigen::VectorXd& x,
        const std::vector<int>& missingIndices,
        const Eigen::VectorXd& halfSpectrum,
        double mu
    ) {
        update(x, missingIndices, halfSpectrum, mu);
    }

    void update(
        const Eigen::VectorXd& x,
        const std::vector<int>& missingIndices,
        const Eigen::VectorXd& halfSpectrum,
        double mu
    ) {
        using Eigen::MatrixXd;
        using Eigen::VectorXd;
        using Eigen::VectorXcd;

        missingIndices_ = missingIndices;

        for (int index : missingIndices_) {
            if (index < 0) throw std::runtime_error("Index < 0");
            if (index >= x.size()) throw std::runtime_error("Index > x.size()");
        }

        int n = x.size();
        int nMissing = missingIndices_.size();

        if (nMissing == 0) return;

        VectorXd fullSpectrum(n);
        int halfSize = halfSpectrum.size();
        fullSpectrum.head(halfSize) = halfSpectrum;
        fullSpectrum.tail(n - halfSize) = halfSpectrum.tail(n - halfSize).reverse();

        VectorXcd fftOutput(n);

        VectorXd xWithMissingZero((x.array() - mu).matrix());
        for (int i = 0; i < missingIndices_.size(); ++i) {
            xWithMissingZero[missingIndices_[i]] = 0;
        }

        fftForward(fftOutput, xWithMissingZero);
        VectorXcd conjFFTXNotMissing = fftOutput.conjugate();
        fftForward(
            fftOutput,
            (conjFFTXNotMissing.array() / fullSpectrum.array()).matrix().eval()
        );
        VectorXd crossCovTimesNotMissing(nMissing);
        for (int i = 0; i < missingIndices_.size(); ++i) {
            crossCovTimesNotMissing[i] = fftOutput[missingIndices_[i]].real() / n;
        }

        fftForward(fftOutput, fullSpectrum.cwiseInverse().eval());
        VectorXd precisions = fftOutput.real() / n;
        MatrixXd missingPrecision(nMissing, nMissing);
        for (int i = 0; i < missingIndices_.size(); ++i) {
            for (int j = 0; j < missingIndices_.size(); ++j) {
                missingPrecision(i, j) = precisions[
                    std::abs(missingIndices_[i] - missingIndices_[j])
                ];
            }
        }

        missingPrecisionLLT_.compute(missingPrecision);
        missingMean_ = (
            -missingPrecisionLLT_.solve(crossCovTimesNotMissing).array()
            + mu
        ).matrix();
    }

    template<typename RNG>
    Eigen::VectorXd operator()(RNG& rng) const {
        if (missingIndices_.size() == 0) {
            return Eigen::VectorXd(0);
        }
        return (
            missingMean_
            + missingPrecisionLLT_.matrixU().solve(randNormal(missingIndices_.size(), rng))
        );
    }

    const std::vector<int>& missingIndices() const {
        return missingIndices_;
    }

private:
    std::vector<int> missingIndices_;
    Eigen::VectorXd missingMean_;
    Eigen::LLT<Eigen::MatrixXd> missingPrecisionLLT_;
};

}  // namespace bayespec

#endif  // SRC_WHITTLE_MISSING_HPP_
