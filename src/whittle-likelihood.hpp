#ifndef SRC_WHITTLE_LIKELIHOOD_HPP_
#define SRC_WHITTLE_LIKELIHOOD_HPP_

namespace bayesspec {

inline
Eigen::VectorXd logWhittleLikelihood(const Eigen::VectorXd& fHat, const Eigen::MatrixXd& periodogram, unsigned int n) {
    unsigned int nHalf = n / 2;
    Eigen::ArrayXd negativeFHatExp = (-fHat).segment(0, nHalf + 1).array().exp();
    Eigen::VectorXd result = Eigen::VectorXd::Zero(periodogram.cols());

    for (unsigned int series = 0; series < periodogram.cols(); ++series) {
        Eigen::VectorXd periodogramExp = (
            periodogram.col(series).segment(0, nHalf + 1).array() * negativeFHatExp
        ).matrix();

        unsigned int nTake = n % 2 == 0 ? nHalf - 1 : nHalf;

        result[series] -= (fHat.segment(1, nTake) + periodogramExp.segment(1, nTake)).sum();
        result[series] -= 0.5 * (fHat[0] + periodogramExp[0]);
        if (n % 2 == 0) {
            result[series] -= 0.5 * (fHat[nHalf] + periodogramExp[nHalf]);
        }
        result[series] -= 0.5 * static_cast<double>(n) * log(2 * M_PI);
    }
    return result;
}

}  // namespace bayespec

#endif  // SRC_WHITTLE_LIKELIHOOD_HPP_
