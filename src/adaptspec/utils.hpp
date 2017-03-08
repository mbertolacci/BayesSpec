#ifndef SRC_ADAPTSPEC_UTILS_HPP_
#define SRC_ADAPTSPEC_UTILS_HPP_

#include <RcppEigen.h>
#include <unsupported/Eigen/FFT>

#include "../splines.hpp"

namespace bayesspec {

class AdaptSpecUtils {
public:
    static
    Eigen::MatrixXd calculateNu(unsigned int n, unsigned int nBases) {
        unsigned int nFrequencies = n / 2;
        return splineBasis1d(
            Eigen::VectorXd::LinSpaced(nFrequencies + 1, 0, 0.5),
            nBases, true
        );
    }

    static
    Eigen::MatrixXd calculatePeriodogram(
        const Eigen::MatrixXd& x,
        unsigned int cutPoint,
        unsigned int n
    ) {
        Eigen::FFT<double> fft;
        Eigen::VectorXcd frequencies;

        unsigned int nRows = n / 2 + 1;

        Eigen::MatrixXd periodogram(nRows, x.cols());
        for (unsigned int series = 0; series < x.cols(); ++series) {
            Eigen::VectorXd thisX = x.col(series).segment(cutPoint - n, n);
            fft.fwd(frequencies, thisX);
            periodogram.col(series) = frequencies.segment(0, nRows).cwiseAbs2()
                / static_cast<double>(n);
        }

        return periodogram;
    }
};

}  // namespace bayesspec

#endif  // SRC_ADAPTSPEC_UTILS_HPP_
