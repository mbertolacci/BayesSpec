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
        periodogram_(periodogram),
        nu_(nu),
        sigmaSquaredAlpha_(sigmaSquaredAlpha),
        tauSquared_(tauSquared) {}

    void setBeta(const Eigen::VectorXd& beta) {
        beta_ = beta;
        fHat_ = nu_ * beta_;

        devExp_ = (
            periodogram_.array().colwise() / fHat_.array().exp()
        ).matrix();

        // Prescale the real-valued special cases
        fHat_[0] *= 0.5;
        devExp_.row(0) = devExp_.row(0) * 0.5;
        if (n_ % 2 == 0) {
            fHat_[n_ / 2] *= 0.5;
            devExp_.row(n_ / 2) = devExp_.row(n_ / 2) * 0.5;
        }
    }

    double value() const {
        // Likelihood contribution
        double result = -periodogram_.cols() * fHat_.sum() - devExp_.sum();

        // Prior contributions
        result -= 0.5 * beta_[0] * beta_[0] / sigmaSquaredAlpha_;
        result -= 0.5 * beta_.segment(1, beta_.size() - 1).squaredNorm() / tauSquared_;

        return -result;
    }

    void gradient(Eigen::VectorXd& gradient) const {
        Eigen::MatrixXd oneMDevExp = (1.0 - devExp_.array()).matrix();
        oneMDevExp.row(0) = (oneMDevExp.row(0).array() - 0.5).matrix();
        if (n_ % 2 == 0) {
            oneMDevExp.row(n_ / 2) = (oneMDevExp.row(n_ / 2).array() - 0.5).matrix();
        }

        // Likelihood contribution
        gradient.fill(0);
        gradient.noalias() -= (oneMDevExp.transpose() * nu_).colwise().sum();

        // Prior contribution
        gradient[0] -= beta_[0] / sigmaSquaredAlpha_;
        gradient.segment(1, beta_.size() - 1).noalias() -= (
            beta_.segment(1, beta_.size() - 1).array() / tauSquared_
        ).matrix();

        gradient *= -1.0;
    }

    void hessian(Eigen::MatrixXd& hessian) const {
        // Likelihood contribution
        hessian.fill(0);
        for (unsigned int series = 0; series < periodogram_.cols(); ++series) {
            hessian.noalias() -= nu_.transpose() * devExp_.col(series).asDiagonal() * nu_;
        }

        // Prior contribution
        hessian(0, 0) -= 1.0 / sigmaSquaredAlpha_;
        for (unsigned int j = 1; j < beta_.size(); ++j) {
            hessian(j, j) -= 1.0 / tauSquared_;
        }

        hessian *= -1.0;
    }

private:
    const unsigned int n_;
    const Eigen::MatrixXd& periodogram_;
    const Eigen::MatrixXd& nu_;
    const double sigmaSquaredAlpha_;
    const double tauSquared_;


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
        lastDirection_.fill(1);
    }

    Status run(Eigen::VectorXd& beta, Eigen::VectorXd& gradient, Eigen::MatrixXd& hessian) {
        currentBeta_ = beta;
        functor_.setBeta(currentBeta_);
        currentValue_ = functor_.value();
        functor_.gradient(currentGradient_);
        functor_.hessian(currentHessian_);
        hessianLLT_.compute(currentHessian_);

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
        // Stop if last step was very small
        if ((lastRate_ * lastDirection_).lpNorm<Eigen::Infinity>() < tolerance_) return CONVERGED;
        if (currentIteration_ == maxIterations_) return MAX_ITERATIONS_REACHED;
        if (hessianLLT_.info() != Eigen::Success) return HESSIAN_NOT_POSITIVE_DEFINITE;
        return RUNNING;
    }

    void takeSingleStep_() {
        lastDirection_ = hessianLLT_.solve(-currentGradient_);

        // Perform back-tracking line-search
        lastRate_ = 1.0;
        double fStart = currentValue_;
        double m = armijoC_ * lastDirection_.dot(currentGradient_);
        functor_.setBeta(currentBeta_ + lastRate_ * lastDirection_);
        currentValue_ = functor_.value();
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
