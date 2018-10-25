#ifndef SRC_ADAPTSPEC_TUNING_HPP_
#define SRC_ADAPTSPEC_TUNING_HPP_

#include <RcppEigen.h>

namespace bayesspec {

class AdaptSpecTuning {
public:
    double probShortMove;
    Eigen::VectorXi shortMoves;
    Eigen::VectorXd shortMoveWeights;
    double varInflate;
    double warmUpVarInflate;
    bool useCutpointWithin;
    bool useSingleWithin;
    bool useHmcWithin;
    int lMin;
    int lMax;
    double epsilonMin;
    double epsilonMax;
    bool useHessianCurvature;

    static AdaptSpecTuning fromList(const Rcpp::List& tuningList) {
        AdaptSpecTuning tuning;
        tuning.probShortMove = tuningList["prob_short_move"];
        tuning.shortMoves = Rcpp::as<Eigen::VectorXi>(tuningList["short_moves"]);
        tuning.shortMoveWeights = Rcpp::as<Eigen::VectorXd>(tuningList["short_move_weights"]);
        tuning.varInflate = tuningList["var_inflate"];
        tuning.warmUpVarInflate = tuningList["warm_up_var_inflate"];
        tuning.useCutpointWithin = tuningList["use_cutpoint_within"];
        tuning.useSingleWithin = tuningList["use_single_within"];
        tuning.useHmcWithin = tuningList["use_hmc_within"];
        tuning.lMin = tuningList["l_min"];
        tuning.lMax = tuningList["l_max"];
        tuning.epsilonMin = tuningList["epsilon_min"];
        tuning.epsilonMax = tuningList["epsilon_max"];
        tuning.useHessianCurvature = tuningList["use_hessian_curvature"];
        return tuning;
    }
};

}  // namespace bayesspec

#endif  // SRC_ADAPTSPEC_TUNING_HPP_
