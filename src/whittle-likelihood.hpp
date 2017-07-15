#ifndef SRC_WHITTLE_LIKELIHOOD_HPP_
#define SRC_WHITTLE_LIKELIHOOD_HPP_

namespace bayesspec {

inline
Eigen::VectorXd logWhittleLikelihood(const Eigen::VectorXd& fHat, const Eigen::MatrixXd& periodogram, unsigned int n) {
    unsigned int nHalf = n / 2;
    unsigned int nTake = n % 2 == 0 ? nHalf - 1 : nHalf;

    Eigen::MatrixXd devExp = (
        periodogram.array().colwise() / fHat.array().exp()
    ).matrix();

    Eigen::VectorXd result(periodogram.cols());

    // Normalising constant
    result.fill(-0.5 * static_cast<double>(n) * log(2 * M_PI));
    // First (scale by 0.5)
    result.array() -= 0.5 * fHat[0];
    result -= 0.5 * devExp.row(0);
    // Middle (scale by 1)
    result.array() -= fHat.segment(1, nTake).sum();
    result -= devExp.block(1, 0, nTake, devExp.cols()).colwise().sum();
    if (n % 2 == 0) {
        // End (scale by 0.5)
        result.array() -= 0.5 * fHat[nHalf];
        result -= 0.5 * devExp.row(nHalf);
    }

    return result;
}

}  // namespace bayespec

#endif  // SRC_WHITTLE_LIKELIHOOD_HPP_
