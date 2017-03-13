#ifndef SRC_RANDOM_TRUNCATED_DISTRIBUTION_HPP_
#define SRC_RANDOM_TRUNCATED_DISTRIBUTION_HPP_

namespace bayesspec {

template<typename Distribution>
class RightTruncatedDistribution {
public:
    typedef typename Distribution::result_type result_type;

    RightTruncatedDistribution(const Distribution& distribution, result_type maximum)
        : distribution_(distribution),
          maximum_(maximum) {}

    template<class RNG>
    result_type operator()(RNG& rng) {
        result_type output;
        while (true) {
            output = distribution_(rng);
            if (output < maximum_) {
                break;
            }
        }
        return output;
    }

private:
    Distribution distribution_;
    result_type maximum_;
};

}  // namespace bayesspec

#endif  // SRC_RANDOM_TRUNCATED_DISTRIBUTION_HPP_
