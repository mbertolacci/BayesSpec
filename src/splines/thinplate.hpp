#ifndef SRC_SPLINES_THINPLATE_HPP_
#define SRC_SPLINES_THINPLATE_HPP_

#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <SymEigsSolver.h>

namespace bayesspec {

class Thinplate2Kernel {
public:
    Thinplate2Kernel() {}

    template<typename InputType>
    Thinplate2Kernel(const Eigen::MatrixBase<InputType>& designMatrix, unsigned int nBases) {
        compute(designMatrix, nBases);
    }

    template<typename InputType>
    Thinplate2Kernel& compute(const Eigen::MatrixBase<InputType>& designMatrix, unsigned int nBases) {
        designMatrix_ = designMatrix;

        normaliseDesignMatrix_();
        if (!computeCovariance_()) {
            return *this;
        }
        computeBasisVectors_(nBases);

        return *this;
    }

    const Eigen::MatrixXd& covariance() const {
        return covariance_;
    }

    const Eigen::MatrixXd& designMatrix() const {
        return designMatrix_;
    }

    const Eigen::VectorXd& eigenvalues() const {
        return eigenvalues_;
    }

    Eigen::ComputationInfo info() const {
        return info_;
    }

private:
    Eigen::MatrixXd designMatrix_;
    Eigen::MatrixXd covariance_;
    Eigen::VectorXd eigenvalues_;

    Eigen::ComputationInfo info_;

    inline void normaliseDesignMatrix_() {
        // Normalise the rows to be between 0 and 1
        Eigen::RowVectorXd minX = designMatrix_.colwise().minCoeff();
        Eigen::RowVectorXd maxX = designMatrix_.colwise().maxCoeff();
        designMatrix_.rowwise() -= minX;
        designMatrix_.array().rowwise() /= (maxX - minX).array();
    }

    inline bool computeCovariance_() {
        unsigned int n = designMatrix_.rows();
        covariance_.resize(n, n);

        // Precalculate values
        Eigen::VectorXd p1(n);
        Eigen::VectorXd p2(n);
        Eigen::VectorXd p3(n);
        Eigen::VectorXd As1(n);
        Eigen::VectorXd As2(n);
        Eigen::VectorXd As3(n);
        #pragma omp parallel for if (n > 5000)
        for (unsigned int i = 0; i < n; ++i) {
            p1[i] = 1.0 - 2.0 * designMatrix_(i, 0) - 2.0 * designMatrix_(i, 1);
            p2[i] = 2.0 * designMatrix_(i, 0);
            p3[i] = 2.0 * designMatrix_(i, 1);
            As1[i] = xLogXApprox_(sqrt(square_(0.00 - designMatrix_(i, 0)) + square_(0.00 - designMatrix_(i, 1))));
            As2[i] = xLogXApprox_(sqrt(square_(0.50 - designMatrix_(i, 0)) + square_(0.00 - designMatrix_(i, 1))));
            As3[i] = xLogXApprox_(sqrt(square_(0.00 - designMatrix_(i, 0)) + square_(0.50 - designMatrix_(i, 1))));
        }

        double As1s2 = xLogXApprox_(0.5);
        double As2s3 = xLogXApprox_(sqrt(0.5));
        double As1s3 = As1s2;

        #pragma omp parallel for if (n > 5000)
        for (unsigned int i = 0; i < n; ++i) {
            for (unsigned int j = i; j < n; ++j) {
                double rij = sqrt(
                    square_(designMatrix_(i, 0) - designMatrix_(j, 0)) +
                    square_(designMatrix_(i, 1) - designMatrix_(j, 1))
                );
                double Aij = xLogXApprox_(rij);

                covariance_(i, j) = (
                    Aij
                    - p1[j] * As1[i] - p2[j] * As2[i] - p3[j] * As3[i] - p1[i] * As1[j] - p2[i] * As2[j] - p3[i] * As3[j]
                    + p1[i] * p2[j] * As1s2 + p1[i] * p3[j] * As1s3 + p2[i] * p1[j] * As1s2 + p2[i] * p3[j] * As2s3
                    + p3[i] * p1[j] * As1s3 + p3[i] * p2[j] * As2s3
                );
                covariance_(j, i) = covariance_(i, j);
            }
        }

        if (!covariance_.allFinite()) {
            info_ = Eigen::NumericalIssue;
            return false;
        }

        return true;
    }

    inline bool computeBasisVectors_(unsigned int nBases) {
        designMatrix_.conservativeResize(Eigen::NoChange, 3 + nBases);
        designMatrix_.col(2) = designMatrix_.col(1);
        designMatrix_.col(1) = designMatrix_.col(0);
        designMatrix_.col(0).setOnes();

        if (nBases == 0) {
            return true;
        }

        unsigned int n = designMatrix_.rows();

        if (nBases == n) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;
            solver.compute(covariance_);

            if (solver.info() != Eigen::Success) {
                info_ = solver.info();
                return false;
            }

            eigenvalues_ = solver.eigenvalues().reverse();
            designMatrix_.rightCols(nBases) = solver.eigenvectors().rowwise().reverse() * eigenvalues_.cwiseSqrt().asDiagonal();
        } else {
            Spectra::DenseSymMatProd<double> op(covariance_);
            Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > solver(
                &op, nBases, std::min(n, 2 * nBases)
            );
            solver.init();
            solver.compute();

            if (solver.info() != Spectra::SUCCESSFUL) {
                info_ = Eigen::NoConvergence;
                return false;
            }

            eigenvalues_ = solver.eigenvalues();
            designMatrix_.rightCols(nBases) = solver.eigenvectors() * eigenvalues_.cwiseSqrt().asDiagonal();
        }

        return true;
    }

    inline double xLogXApprox_(double x) const {
        double result = x * log(x);
        if (!std::isfinite(result)) {
            return 0;
        }
        return result;
    }

    inline double square_(double x) const { return x * x; }
};

}  // namespace bayesspec

#endif  // SRC_SPLINES_THINPLATE_HPP_
