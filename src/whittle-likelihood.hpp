#ifndef SRC_WHITTLE_LIKELIHOOD_HPP_
#define SRC_WHITTLE_LIKELIHOOD_HPP_

namespace bayesspec {

inline
Eigen::VectorXd logWhittleLikelihood(const Eigen::VectorXd& fHat, const Eigen::MatrixXd& periodogram, unsigned int n) {
    unsigned int nHalf = n / 2;
    unsigned int nTake = n % 2 == 0 ? nHalf - 1 : nHalf;

    Eigen::MatrixXd devExp = (
        periodogram.topRows(nHalf + 1).array().log().colwise() - fHat.array()
    ).exp().matrix();

    Eigen::VectorXd result = Eigen::VectorXd::Zero(periodogram.cols());
    for (unsigned int series = 0; series < periodogram.cols(); ++series) {
        result[series] -= (fHat.segment(1, nTake) + devExp.col(series).segment(1, nTake)).sum();
        result[series] -= 0.5 * (fHat[0] + devExp(0, series));
        if (n % 2 == 0) {
            result[series] -= 0.5 * (fHat[nHalf] + devExp(nHalf, series));
        }
        result[series] -= 0.5 * static_cast<double>(n) * log(2 * M_PI);
    }
    return result;
}

}  // namespace bayespec

#endif  // SRC_WHITTLE_LIKELIHOOD_HPP_
