#ifndef SRC_TRUNCATED_INVERSE_GAMMA_HPP_
#define SRC_TRUNCATED_INVERSE_GAMMA_HPP_

#include "rng.hpp"

namespace bayesspec {

inline
double rTruncatedInverseGamma(double alpha, double beta, double maximum) {
    double output;

    while (true) {
        output = beta / rng.randg(alpha, 1);

        if (output < maximum) {
            break;
        }
    }

    return output;
}

}  // namespace bayesspec

#endif  // SRC_TRUNCATED_INVERSE_GAMMA_HPP_
