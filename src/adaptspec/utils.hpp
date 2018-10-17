#ifndef SRC_ADAPTSPEC_UTILS_HPP_
#define SRC_ADAPTSPEC_UTILS_HPP_

#include <RcppEigen.h>

#ifdef BAYESSPEC_FFTW
    #include <fftw3.h>
#else
    #include <unsupported/Eigen/FFT>
#endif

#include "../splines.hpp"

namespace bayesspec {

class AdaptSpecUtils {
public:
    static
    Eigen::MatrixXd calculateNu(unsigned int n, unsigned int nBases, bool cubeRoot = false) {
        unsigned int maxFrequency = n / 2;
        Eigen::VectorXd frequencies = (
            Eigen::VectorXd::LinSpaced(maxFrequency + 1, 0, maxFrequency)
            / static_cast<double>(n)
        );
        if (cubeRoot) {
            for (int i = 0; i < frequencies.size(); ++i) {
                frequencies[i] = std::cbrt(frequencies[i]);
            }
        }
        return splineBasis1dDemmlerReinsch(
            frequencies,
            nBases
        );
    }

    static
    Eigen::MatrixXd calculatePeriodogram(
        const Eigen::MatrixXd& x,
        unsigned int cutPoint,
        unsigned int n
    ) {
        unsigned int nRows = n / 2 + 1;

        Eigen::VectorXd thisX(n);
        Eigen::VectorXcd frequencies(nRows);

        #ifdef BAYESSPEC_FFTW
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
        #else
            Eigen::FFT<double> fft;
        #endif

        Eigen::MatrixXd periodogram(nRows, x.cols());
        for (unsigned int series = 0; series < x.cols(); ++series) {
            thisX = x.col(series).segment(cutPoint - n, n);
            #ifdef BAYESSPEC_FFTW
                fftw_execute(plan);
            #else
                fft.fwd(frequencies, thisX);
            #endif
            periodogram.col(series) = frequencies.cwiseAbs2() / static_cast<double>(n);
        }

        #ifdef BAYESSPEC_FFTW
            #pragma omp critical
            {
                fftw_destroy_plan(plan);
            }
        #endif

        return periodogram;
    }
};

}  // namespace bayesspec

#endif  // SRC_ADAPTSPEC_UTILS_HPP_
