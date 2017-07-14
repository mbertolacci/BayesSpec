#ifndef SRC_ADAPTSPEC_SEGMENT_FUNCTOR_HPP_
#define SRC_ADAPTSPEC_SEGMENT_FUNCTOR_HPP_

namespace bayesspec {

class BetaFunctor {
public:
    BetaFunctor(
        unsigned int n,
        const Eigen::MatrixXd& periodogram,
        const Eigen::MatrixXd& nu,
        double sigmaSquaredAlpha,
        double tauSquared
    ) : n_(n),
        logPeriodogram_(periodogram.topRows(n / 2 + 1).array().log()),
        nu_(nu),
        sigmaSquaredAlpha_(sigmaSquaredAlpha),
        tauSquared_(tauSquared) {}

    void setBeta(const Eigen::VectorXd& beta) {
        beta_ = beta;
        fHat_ = nu_ * beta_;
        devExp_ = (
            logPeriodogram_.colwise() - fHat_.array()
        ).exp().matrix();
    }

    double value() const {
        unsigned int nHalf = n_ / 2;
        unsigned int nTake = n_ % 2 == 0 ? nHalf - 1 : nHalf;

        double result = 0;
        for (unsigned int series = 0; series < logPeriodogram_.cols(); ++series) {
            result -= (fHat_.segment(1, nTake) + devExp_.col(series).segment(1, nTake)).sum();
            result -= 0.5 * (fHat_[0] + devExp_(0, series));
            if (n_ % 2 == 0) {
                result -= 0.5 * (fHat_[nHalf] + devExp_(nHalf, series));
            }
            result -= 0.5 * static_cast<double>(n_) * log(2 * M_PI);
        }

        result -= 0.5 * beta_[0] * beta_[0] / sigmaSquaredAlpha_;
        result -= 0.5 * beta_.segment(1, beta_.size() - 1).squaredNorm() / tauSquared_;

        return -result;
    }

    void gradient(Eigen::VectorXd& gradient) const {
        unsigned int nHalf = n_ / 2;
        Eigen::MatrixXd oneMDevExp = (1.0 - devExp_.array()).matrix();

        // Likelihood contribution
        gradient.fill(0);
        for (unsigned int series = 0; series < logPeriodogram_.cols(); ++series) {
            if (n_ % 2 == 1) {
                // Odd
                gradient += -nu_.block(1, 0, nHalf, nu_.cols()).transpose() * oneMDevExp.col(series).segment(1, nHalf);
            } else {
                // Even
                gradient += -nu_.block(1, 0, nHalf - 1, nu_.cols()).transpose() * oneMDevExp.col(series).segment(1, nHalf - 1)
                    - 0.5 * oneMDevExp(nHalf, series) * nu_.row(nHalf).transpose();
            }
            gradient -= 0.5 * oneMDevExp(0, series) * nu_.row(0).transpose();
        }

        // Prior contribution
        gradient[0] -= beta_[0] / sigmaSquaredAlpha_;
        gradient.segment(1, beta_.size() - 1) -= (
            beta_.segment(1, beta_.size() - 1).array() / tauSquared_
        ).matrix();

        gradient *= -1.0;
    }

    void hessian(Eigen::MatrixXd& hessian) const {
        unsigned int nHalf = n_ / 2;

        // Likelihood contribution
        hessian.fill(0);
        for (unsigned int series = 0; series < logPeriodogram_.cols(); ++series) {
            if (n_ % 2 == 1) {
                // Odd
                hessian += (
                    -nu_.block(1, 0, nHalf, nu_.cols()).transpose()
                    * devExp_.col(series).segment(1, nHalf).asDiagonal()
                    * nu_.block(1, 0, nHalf, nu_.cols())
                ) - 0.5 * devExp_(0, series) * nu_.row(0).transpose() * nu_.row(0);
            } else {
                // Even
                hessian += (
                    -nu_.block(1, 0, nHalf - 1, nu_.cols()).transpose()
                    * devExp_.col(series).segment(1, nHalf - 1).asDiagonal()
                    * nu_.block(1, 0, nHalf - 1, nu_.cols())
                ) - 0.5 * devExp_(0, series) * nu_.row(0).transpose() * nu_.row(0)
                  - 0.5 * devExp_(nHalf, series) * nu_.row(nHalf).transpose() * nu_.row(nHalf);
            }
        }

        // Prior contribution
        hessian(0, 0) -= 1.0 / sigmaSquaredAlpha_;
        for (unsigned int j = 1; j < beta_.size(); ++j) {
            hessian(j, j) -= 1.0 / tauSquared_;
        }

        hessian *= -1.0;
    }

    void reasonableStart(Eigen::VectorXd& beta) const {
        // NOTE(mgnb): Numerically this is a safe starting point, because it
        // makes the log spectrum a constant that bounds the log periodogram
        // from above
        beta.fill(0);
        if (logPeriodogram_.size() > 0) {
            beta[0] = logPeriodogram_.maxCoeff();
        }
    }

private:
    const unsigned int n_;
    const Eigen::MatrixXd nu_;
    const double sigmaSquaredAlpha_;
    const double tauSquared_;
    const Eigen::ArrayXXd logPeriodogram_;

    // Mutables
    Eigen::VectorXd beta_;
    Eigen::VectorXd fHat_;
    Eigen::MatrixXd devExp_;
};

class BetaOptimiser {
public:
    enum Status {
        RUNNING = 0,
        CONVERGED = 1,
        MAX_ITERATIONS_REACHED = 2,
        HESSIAN_NOT_POSITIVE_DEFINITE = 3
    };

    BetaOptimiser(
        unsigned int n,
        const Eigen::MatrixXd& periodogram,
        const Eigen::MatrixXd& nu,
        double sigmaSquaredAlpha,
        double tauSquared
    ) : functor_(n, periodogram, nu, sigmaSquaredAlpha, tauSquared),
        currentIteration_(0),
        currentBeta_(nu.cols()),
        currentGradient_(nu.cols()),
        currentHessian_(nu.cols(), nu.cols()),
        lastRate_(1.0),
        lastDirection_(nu.cols()),
        tolerance_(sqrtEpsilon()),
        armijoC_(0.01),
        armijoRho_(0.5) {
        functor_.reasonableStart(currentBeta_);
        functor_.setBeta(currentBeta_);
        functor_.gradient(currentGradient_);
        functor_.hessian(currentHessian_);
        hessianLLT_.compute(currentHessian_);
        lastDirection_.fill(1);
    }

    Status run(Eigen::VectorXd& beta, Eigen::VectorXd& gradient, Eigen::MatrixXd& hessian) {
        while (status_() == RUNNING) {
            takeSingleStep_();
        }

        beta = currentBeta_;
        gradient = currentGradient_;
        hessian = currentHessian_;
        return status_();
    }

    friend std::ostream& operator<< (std::ostream& stream, const BetaOptimiser& optimiser) {
        stream << "currentIteration_ = " << optimiser.currentIteration_ << "\n";
        stream << "currentBeta_ =\n" << optimiser.currentBeta_.transpose() << "\n";
        stream << "currentValue_ = " << optimiser.currentValue_ << "\n";
        stream << "currentGradient_ =\n" << optimiser.currentGradient_.transpose() << "\n";
        stream << "currentHessian_ =\n" << optimiser.currentHessian_ << "\n";
        stream << "tolerance_ = " << optimiser.tolerance_ << "\n";
        stream << "lastDirection_ =\n" << optimiser.lastDirection_.transpose() << "\n";
        stream << "lastRate_ = " << optimiser.lastRate_ << "\n";
        stream << "status_() = " << optimiser.status_();
        return stream;
    }

private:
    static const unsigned int maxIterations_ = 1000;

    BetaFunctor functor_;
    unsigned int currentIteration_;
    Eigen::VectorXd currentBeta_;
    double currentValue_;
    Eigen::VectorXd currentGradient_;
    Eigen::MatrixXd currentHessian_;
    Eigen::LLT<Eigen::MatrixXd> hessianLLT_;
    double lastRate_;
    Eigen::VectorXd lastDirection_;

    double tolerance_;
    double armijoC_;
    double armijoRho_;

    Status status_() const {
        // Stop if gradient very small
        if (currentGradient_.lpNorm<Eigen::Infinity>() < tolerance_) return CONVERGED;
        // Stop is last step was very small
        if ((lastRate_ * lastDirection_).lpNorm<Eigen::Infinity>() < tolerance_) return CONVERGED;
        if (currentIteration_ == maxIterations_) return MAX_ITERATIONS_REACHED;
        if (hessianLLT_.info() != Eigen::Success) return HESSIAN_NOT_POSITIVE_DEFINITE;
        return RUNNING;
    }

    void takeSingleStep_() {
        lastDirection_ = hessianLLT_.solve(-currentGradient_);

        // Perform back-tracking line-search
        lastRate_ = 1.0;
        double fStart = functor_.value();
        double m = armijoC_ * lastDirection_.dot(currentGradient_);
        functor_.setBeta(currentBeta_ + lastRate_ * lastDirection_);
        double currentValue_ = functor_.value();
        while (
            (lastRate_ * lastDirection_).lpNorm<Eigen::Infinity>() > tolerance_
            && currentValue_ - fStart > lastRate_ * m
        ) {
            lastRate_ *= armijoRho_;
            functor_.setBeta(currentBeta_ + lastRate_ * lastDirection_);
            currentValue_ = functor_.value();
        }

        // Set the new beta and update gradient and hessian
        currentBeta_ += lastRate_ * lastDirection_;
        functor_.gradient(currentGradient_);
        functor_.hessian(currentHessian_);
        hessianLLT_.compute(currentHessian_);

        ++currentIteration_;
    }

    // Sqrt machine precision
    static double sqrtEpsilon() {
      return std::sqrt(Eigen::NumTraits<double>::epsilon());
    }
};

}  // namespace bayespec

#endif  // SRC_ADAPTSPEC_SEGMENT_FUNCTOR_HPP_
