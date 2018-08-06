#include "utils.hpp"

namespace bayesspec {

Rcpp::List missingValuesAsList(
    const Eigen::MatrixXd& x,
    const std::vector<Eigen::VectorXi>& missingIndices
) {
    Rcpp::List output;
    for (int i = 0; i < missingIndices.size(); ++i) {
        Rcpp::NumericVector xMissing(missingIndices[i].size());
        for (int j = 0; j < missingIndices[i].size(); ++j) {
            xMissing[j] = x(missingIndices[i][j], i);
        }
        output.push_back(xMissing);
    }
    return output;
}

}  // namespace bayesspec
