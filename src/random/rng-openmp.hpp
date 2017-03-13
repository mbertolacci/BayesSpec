#ifndef SRC_RANDOM_RNG_OPENMP_HPP_
#define SRC_RANDOM_RNG_OPENMP_HPP_

#include <omp.h>

namespace bayesspec {

template<typename Engine>
class RNGOpenMP {
public:
    typedef typename Engine::result_type result_type;

    template<typename Lambda>
    explicit RNGOpenMP(Lambda getSeed) {
        unsigned int threads = std::max(1, omp_get_max_threads());

        for (unsigned int thread = 0; thread < threads; ++thread) {
            engines_.emplace_back(getSeed());
        }
    }

    result_type operator()() {
        return engines_[omp_get_thread_num()]();
    }

    result_type min() {
        return engines_[0].min();
    }

    result_type max() {
        return engines_[0].max();
    }

private:
    std::vector<Engine> engines_;
};

}  // namespace bayesspec

#endif  // SRC_RANDOM_RNG_OPENMP_HPP_
