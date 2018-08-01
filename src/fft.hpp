#ifndef SRC_FFT_HPP_
#define SRC_FFT_HPP_

#include <RcppEigen.h>

#ifdef BAYESSPEC_FFTW
    #include <fftw3.h>
#else
    #include <unsupported/Eigen/FFT>
#endif

namespace bayesspec {

// Real to complex

inline
void fftForward(Eigen::VectorXcd& destination, const Eigen::VectorXd& source) {
    #ifdef BAYESSPEC_FFTW
        int n = static_cast<int>(source.size());
        fftw_plan plan;
        #pragma omp critical
        {
            plan = fftw_plan_dft_r2c_1d(
                n,
                const_cast<double *>(source.data()),
                reinterpret_cast<fftw_complex *>(destination.data()),
                FFTW_ESTIMATE | FFTW_PRESERVE_INPUT
            );
        }
    #else
        Eigen::FFT<double> fft;
    #endif

    #ifdef BAYESSPEC_FFTW
        fftw_execute(plan);
        int nBins = n / 2 + 1;
        for (int i = nBins; i < n; ++i) {
            destination[i] = std::conj(destination[n - i]);
        }
    #else
        fft.fwd(destination, source);
    #endif

    #ifdef BAYESSPEC_FFTW
        #pragma omp critical
        {
            fftw_destroy_plan(plan);
        }
    #endif
}

inline
Eigen::VectorXcd fftForward(const Eigen::VectorXd& source) {
    Eigen::VectorXcd destination(source.size());
    fftForward(destination, source);
    return destination;
}

// Complex to complex

inline
void fftForward(Eigen::VectorXcd& destination, const Eigen::VectorXcd& source) {
    #ifdef BAYESSPEC_FFTW
        fftw_plan plan;
        #pragma omp critical
        {
            plan = fftw_plan_dft_1d(
                static_cast<int>(source.size()),
                reinterpret_cast<fftw_complex *>(const_cast<std::complex<double> *>(source.data())),
                reinterpret_cast<fftw_complex *>(destination.data()),
                FFTW_FORWARD,
                FFTW_ESTIMATE | FFTW_PRESERVE_INPUT
            );
        }
    #else
        Eigen::FFT<double> fft;
    #endif

    #ifdef BAYESSPEC_FFTW
        fftw_execute(plan);
    #else
        fft.fwd(destination, source);
    #endif

    #ifdef BAYESSPEC_FFTW
        #pragma omp critical
        {
            fftw_destroy_plan(plan);
        }
    #endif
}

inline
Eigen::VectorXcd fftForward(const Eigen::VectorXcd& source) {
    Eigen::VectorXcd destination(source.size());
    fftForward(destination, source);
    return destination;
}

}  // namespace bayesspec

#endif  // SRC_FFT_HPP_
