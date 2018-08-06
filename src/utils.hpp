#ifndef SRC_UTILS_HPP_
#define SRC_UTILS_HPP_

#include <RcppEigen.h>

namespace bayesspec {

Rcpp::List missingValuesAsList(
    const Eigen::MatrixXd& x,
    const std::vector<Eigen::VectorXi>& missingIndices
);

}  // namespace bayesspec

#endif  // SRC_UTILS_HPP_
