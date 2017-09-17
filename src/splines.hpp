#ifndef SRC_SPLINES_HPP_
#define SRC_SPLINES_HPP_

#include <RcppEigen.h>

namespace bayesspec {

// Assumes that x is in [0, 1] already
template<typename VectorType>
Eigen::MatrixXd splineBasis1dDemmlerReinsch(const VectorType& x, unsigned int nBases) {
    Eigen::MatrixXd designMatrix(x.size(), nBases + 1);

    designMatrix.col(0).fill(1);

    for (unsigned int j = 0; j < nBases; ++j) {
        double jF = static_cast<double>(j) + 1.0;
        designMatrix.col(j + 1) = sqrt(2.0) * (jF * M_PI * x.array()).cos() / (M_PI * jF);
    }

    return designMatrix;
}

}  // namespace bayesspec

#endif  // SRC_SPLINES_HPP_
