#ifndef SRC_ADAPTSPEC_UTILS_HPP_
#define SRC_ADAPTSPEC_UTILS_HPP_

#include <RcppEigen.h>
// #include <unsupported/Eigen/FFT>
#include <fftw3.h>

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
        // Eigen::FFT<double> fft;

        unsigned int nRows = n / 2 + 1;

        Eigen::VectorXd thisX(n);
        Eigen::VectorXcd frequencies(nRows);

        fftw_plan plan;
        #pragma omp critical
        {
            plan = fftw_plan_dft_r2c_1d(
                n,
                thisX.data(),
                reinterpret_cast<fftw_complex *>(frequencies.data()),
                FFTW_ESTIMATE
            );
        }

        Eigen::MatrixXd periodogram(nRows, x.cols());
        for (unsigned int series = 0; series < x.cols(); ++series) {
            thisX = x.col(series).segment(cutPoint - n, n);
            fftw_execute(plan);
            periodogram.col(series) = frequencies.cwiseAbs2() / static_cast<double>(n);
        }

        #pragma omp critical
        {
            fftw_destroy_plan(plan);
        }

        return periodogram;
    }
};

}  // namespace bayesspec

#endif  // SRC_ADAPTSPEC_UTILS_HPP_
