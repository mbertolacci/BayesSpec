#ifndef SRC_ADAPTSPEC_TUNING_HPP_
#define SRC_ADAPTSPEC_TUNING_HPP_

namespace bayesspec {

class AdaptSpecTuning {
public:
    double probShortMove;
    double varInflate;
    double warmUpVarInflate;

    static AdaptSpecTuning fromList(const Rcpp::List& tuningList) {
        AdaptSpecTuning tuning;
        tuning.probShortMove = tuningList["prob_short_move"];
        tuning.varInflate = tuningList["var_inflate"];
        tuning.warmUpVarInflate = tuningList["warm_up_var_inflate"];
        return tuning;
    }
};

}  // namespace bayesspec

#endif  // SRC_ADAPTSPEC_TUNING_HPP_
