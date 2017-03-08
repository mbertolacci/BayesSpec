#ifndef SRC_ADAPTSPEC_SEGMENT_FUNCTOR_HPP_
#define SRC_ADAPTSPEC_SEGMENT_FUNCTOR_HPP_

#include "../cppoptlib/problem.h"

namespace bayesspec {

class SegmentFunctor : public cppoptlib::Problem<double> {
public:
    SegmentFunctor(
        unsigned int n,
        const Eigen::MatrixXd& periodogram,
        const Eigen::MatrixXd& nu,
        double sigmaSquaredAlpha,
        double tauSquared
    ) : n_(n),
        periodogram_(periodogram),
        nu_(nu),
        sigmaSquaredAlpha_(sigmaSquaredAlpha),
        tauSquared_(tauSquared) {};

    double value(const Eigen::VectorXd& beta) {
        unsigned int nHalf = n_ / 2;
        Eigen::VectorXd fHat = nu_ * beta;
        Eigen::ArrayXd negativeFHatExp = (-fHat).segment(0, nHalf + 1).array().exp();
        unsigned int nTake = n_ % 2 == 0 ? nHalf - 1 : nHalf;

        double result = 0;
        for (unsigned int series = 0; series < periodogram_.cols(); ++series) {
            Eigen::VectorXd periodogramExp = (
                periodogram_.col(series).segment(0, nHalf + 1).array() * negativeFHatExp
            ).matrix();

            result -= (fHat.segment(1, nTake) + periodogramExp.segment(1, nTake)).sum();
            result -= 0.5 * (fHat[0] + periodogramExp[0]);
            if (n_ % 2 == 0) {
                result -= 0.5 * (fHat[nHalf] + periodogramExp[nHalf]);
            }
            result -= 0.5 * static_cast<double>(n_) * log(2 * M_PI);
        }

        result -= 0.5 * beta[0] * beta[0] / sigmaSquaredAlpha_;
        result -= 0.5 * beta.segment(1, beta.size() - 1).squaredNorm() / tauSquared_;

        return -result;
    }

    void gradient(const Eigen::VectorXd& beta, Eigen::VectorXd& gradient) {
        unsigned int nHalf = n_ / 2;
        Eigen::ArrayXd negativeFHatExp = (-nu_ * beta).segment(0, nHalf + 1).array().exp();

        gradient.fill(0);
        for (unsigned int series = 0; series < periodogram_.cols(); ++series) {
            Eigen::VectorXd oneMPeriodogramExp = (
                1.0 - periodogram_.col(series).segment(0, nHalf + 1).array() * negativeFHatExp
            ).matrix();
            if (n_ % 2 == 1) {
                // Odd
                gradient += -nu_.block(1, 0, nHalf, nu_.cols()).transpose() * oneMPeriodogramExp.segment(1, nHalf);
            } else {
                // Even
                gradient += -nu_.block(1, 0, nHalf - 1, nu_.cols()).transpose() * oneMPeriodogramExp.segment(1, nHalf - 1)
                    - 0.5 * oneMPeriodogramExp[nHalf] * nu_.row(nHalf).transpose();
            }
            gradient -= 0.5 * oneMPeriodogramExp[0] * nu_.row(0).transpose();
        }
        gradient[0] -= beta[0] / sigmaSquaredAlpha_;
        gradient.segment(1, beta.size() - 1) -= (
            beta.segment(1, beta.size() - 1).array() / tauSquared_
        ).matrix();

        gradient *= -1.0;
    }

    void hessian(const Eigen::VectorXd& beta, Eigen::MatrixXd& hessian) {
        unsigned int nHalf = n_ / 2;
        Eigen::ArrayXd negativeFHatExp = (-nu_ * beta).segment(0, nHalf + 1).array().exp();
        Eigen::VectorXd periodogramExp(negativeFHatExp.size());

        hessian.fill(0);
        for (unsigned int series = 0; series < periodogram_.cols(); ++series) {
            periodogramExp = (periodogram_.col(series).segment(0, nHalf + 1).array() * negativeFHatExp).matrix();

            if (n_ % 2 == 1) {
                // Odd
                hessian += (
                    -nu_.block(1, 0, nHalf, nu_.cols()).transpose()
                    * periodogramExp.segment(1, nHalf).asDiagonal()
                    * nu_.block(1, 0, nHalf, nu_.cols())
                ) - 0.5 * periodogramExp[0] * nu_.row(0).transpose() * nu_.row(0);
            } else {
                // Even
                hessian += (
                    -nu_.block(1, 0, nHalf - 1, nu_.cols()).transpose()
                    * periodogramExp.segment(1, nHalf - 1).asDiagonal()
                    * nu_.block(1, 0, nHalf - 1, nu_.cols())
                ) - 0.5 * periodogramExp[0] * nu_.row(0).transpose() * nu_.row(0)
                  - 0.5 * periodogramExp[nHalf] * nu_.row(nHalf).transpose() * nu_.row(nHalf);
            }
        }

        hessian(0, 0) -= 1.0 / sigmaSquaredAlpha_;
        for (unsigned int j = 1; j < beta.size(); ++j) {
            hessian(j, j) -= 1.0 / tauSquared_;
        }

        hessian *= -1.0;
    }

private:
    unsigned int n_;
    const Eigen::MatrixXd& periodogram_;
    const Eigen::MatrixXd& nu_;
    double sigmaSquaredAlpha_;
    double tauSquared_;
};

}  // namespace bayespec

#endif  // SRC_ADAPTSPEC_SEGMENT_FUNCTOR_HPP_
