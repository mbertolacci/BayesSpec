#ifndef SRC_WHITTLE_LIKELIHOOD_HPP_
#define SRC_WHITTLE_LIKELIHOOD_HPP_

namespace bayesspec {

inline
double logWhittleLikelihood(const Eigen::VectorXd& fHat, const Eigen::VectorXd& periodogram, unsigned int n) {
    unsigned int nHalf = n / 2;

    Eigen::VectorXd periodogramExp = (
        periodogram.segment(0, nHalf + 1).array()
        * (-fHat).segment(0, nHalf + 1).array().exp()
    ).matrix();

    unsigned int nTake = n % 2 == 0 ? nHalf - 1 : nHalf;

    double result = 0;
    result -= (fHat.segment(1, nTake) + periodogramExp.segment(1, nTake)).sum();
    result -= 0.5 * (fHat[0] + periodogramExp[0]);
    if (n % 2 == 0) {
        result -= 0.5 * (fHat[nHalf] + periodogramExp[nHalf]);
    }
    result -= 0.5 * static_cast<double>(n) * log(2 * M_PI);
    return result;
}

}  // namespace bayespec

#endif  // SRC_WHITTLE_LIKELIHOOD_HPP_
