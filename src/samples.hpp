#ifndef SRC_SAMPLES_HPP_
#define SRC_SAMPLES_HPP_

#include <algorithm>
#include <vector>
#include <functional>
#include <cassert>

namespace bayesspec {

template<typename T>
class Samples {
public:
    typedef std::vector<std::size_t> Dimensions;
    typedef std::vector<T> SamplesStorage;

    Samples(
        size_t nSamples,
        size_t thin,
        const Dimensions& dimensions,
        bool forceMatrix = false
    )
        : currentIndex_(0),
          thin_(thin),
          dimensions_(dimensions),
          forceMatrix_(forceMatrix),
          size_(std::accumulate(
            dimensions.cbegin(), dimensions.cend(), 1, std::multiplies<int>()
          )),
          samples_(
            thin_ == 0 ? 0 : size_ * static_cast<std::size_t>(std::ceil(
                static_cast<double>(nSamples) / static_cast<double>(thin)
            ))
          ) {}

    Samples(size_t nSamples, size_t thin, size_t length, bool forceMatrix = false)
        : Samples(
            nSamples,
            thin,
            Dimensions({ static_cast<size_t>(length) }),
            forceMatrix
        ) {}

    Samples(size_t nSamples, size_t thin)
        : Samples(
            nSamples,
            thin,
            Dimensions({ static_cast<size_t>(1) })
        ) {}

    void save(const T& input) {
        assert(size_ == 1);
        if (thin_ > 0 && currentIndex_ % thin_ == 0) {
            samples_[currentIndex_ / thin_] = input;
        }
        ++currentIndex_;
    }

    template<typename Input>
    void save(const Input& input) {
        assert(size_ == input.size());
        if (thin_ > 0 && currentIndex_ % thin_ == 0) {
            std::copy(
                input.data(),
                input.data() + size_,
                samples_.begin() + (currentIndex_ / thin_) * size_
            );
        }
        ++currentIndex_;
    }

    typename SamplesStorage::const_iterator cbegin() const {
        return samples_.cbegin();
    }

    typename SamplesStorage::const_iterator cend() const {
        return samples_.cend();
    }

    const Dimensions& dimensions() const {
        return dimensions_;
    }

    bool forceMatrix() const {
        return forceMatrix_;
    }

    size_t nStoredSamples() const {
        return samples_.size() == 0 ? 0 : samples_.size() / size_;
    }

    size_t thin() const {
        return thin_;
    }

private:
    size_t currentIndex_;
    size_t thin_;
    Dimensions dimensions_;
    bool forceMatrix_;
    size_t size_;
    SamplesStorage samples_;
};

}

namespace Rcpp {

template <typename T>
SEXP wrap(const bayesspec::Samples<T>& input) {
    const int RTYPE = Rcpp::traits::r_sexptype_traits<T>::rtype;

    if (input.nStoredSamples() == 0) {
        return R_NilValue;
    }

    Rcpp::IntegerVector mcpar(Rcpp::IntegerVector({
        1,
        static_cast<int>(1 + (input.nStoredSamples() - 1) * input.thin()),
        static_cast<int>(input.thin())
    }));

    if (input.dimensions().size() == 1 && (input.dimensions()[0] > 1 || input.forceMatrix())) {
        Rcpp::Matrix<RTYPE> output(
            input.dimensions()[0],
            input.nStoredSamples()
        );
        std::copy(input.cbegin(), input.cend(), output.begin());
        output = Rcpp::transpose(output);
        output.attr("class") = "mcmc";
        output.attr("mcpar") = mcpar;
        return output;
    } else {
        Rcpp::Vector<RTYPE> output(input.cbegin(), input.cend());

        if (input.dimensions().size() == 1) {
            output.attr("class") = "mcmc";
        } else {
            Rcpp::IntegerVector dimensions(0);
            for (std::size_t d : input.dimensions()) {
                dimensions.push_back(static_cast<int>(d));
            }
            dimensions.push_back(static_cast<int>(input.nStoredSamples()));
            output.attr("dim") = dimensions;

            Rcpp::IntegerVector permutation(0);
            permutation.push_back(dimensions.size());
            for (int i = 1; i < dimensions.size(); ++i) {
                permutation.push_back(i);
            }
            Rcpp::Function aperm = Environment::base_env()["aperm"];
            output = aperm(output, permutation);
            output.attr("class") = "mcmca";
        }

        output.attr("mcpar") = mcpar;
        return output;
    }
}

}

#endif  // SRC_SAMPLES_HPP_
