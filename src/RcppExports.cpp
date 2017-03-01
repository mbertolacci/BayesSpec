// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// adaptspec
Rcpp::List adaptspec(unsigned int nLoop, unsigned int nWarmUp, Rcpp::NumericMatrix xR, Rcpp::List priorList, double probMM1, unsigned int nSegmentsStart, bool showProgress);
RcppExport SEXP BayesSpec_adaptspec(SEXP nLoopSEXP, SEXP nWarmUpSEXP, SEXP xRSEXP, SEXP priorListSEXP, SEXP probMM1SEXP, SEXP nSegmentsStartSEXP, SEXP showProgressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type nLoop(nLoopSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nWarmUp(nWarmUpSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type priorList(priorListSEXP);
    Rcpp::traits::input_parameter< double >::type probMM1(probMM1SEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nSegmentsStart(nSegmentsStartSEXP);
    Rcpp::traits::input_parameter< bool >::type showProgress(showProgressSEXP);
    rcpp_result_gen = Rcpp::wrap(adaptspec(nLoop, nWarmUp, xR, priorList, probMM1, nSegmentsStart, showProgress));
    return rcpp_result_gen;
END_RCPP
}
// independentMixture
Rcpp::List independentMixture();
RcppExport SEXP BayesSpec_independentMixture() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(independentMixture());
    return rcpp_result_gen;
END_RCPP
}
// getSampleDefault
Rcpp::List getSampleDefault(Rcpp::NumericMatrix xR, Rcpp::List priorList, unsigned int nStartingSegments);
RcppExport SEXP BayesSpec_getSampleDefault(SEXP xRSEXP, SEXP priorListSEXP, SEXP nStartingSegmentsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type priorList(priorListSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nStartingSegments(nStartingSegmentsSEXP);
    rcpp_result_gen = Rcpp::wrap(getSampleDefault(xR, priorList, nStartingSegments));
    return rcpp_result_gen;
END_RCPP
}
// getSampleFilled
Rcpp::List getSampleFilled(Rcpp::NumericMatrix xR, Rcpp::List priorList, unsigned int nSegments, Rcpp::NumericMatrix beta, Rcpp::NumericVector tauSquared, Rcpp::IntegerVector cutPoints);
RcppExport SEXP BayesSpec_getSampleFilled(SEXP xRSEXP, SEXP priorListSEXP, SEXP nSegmentsSEXP, SEXP betaSEXP, SEXP tauSquaredSEXP, SEXP cutPointsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type priorList(priorListSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nSegments(nSegmentsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type tauSquared(tauSquaredSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type cutPoints(cutPointsSEXP);
    rcpp_result_gen = Rcpp::wrap(getSampleFilled(xR, priorList, nSegments, beta, tauSquared, cutPoints));
    return rcpp_result_gen;
END_RCPP
}
// getMetropolisLogRatio
double getMetropolisLogRatio(Rcpp::List currentR, Rcpp::List proposalR, Rcpp::NumericMatrix xR, Rcpp::List priorList);
RcppExport SEXP BayesSpec_getMetropolisLogRatio(SEXP currentRSEXP, SEXP proposalRSEXP, SEXP xRSEXP, SEXP priorListSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type currentR(currentRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type proposalR(proposalRSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type priorList(priorListSEXP);
    rcpp_result_gen = Rcpp::wrap(getMetropolisLogRatio(currentR, proposalR, xR, priorList));
    return rcpp_result_gen;
END_RCPP
}
// splineBasis1dR
Rcpp::NumericMatrix splineBasis1dR(Rcpp::NumericVector xR, unsigned int nBases, bool omitLinear);
RcppExport SEXP BayesSpec_splineBasis1dR(SEXP xRSEXP, SEXP nBasesSEXP, SEXP omitLinearSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nBases(nBasesSEXP);
    Rcpp::traits::input_parameter< bool >::type omitLinear(omitLinearSEXP);
    rcpp_result_gen = Rcpp::wrap(splineBasis1dR(xR, nBases, omitLinear));
    return rcpp_result_gen;
END_RCPP
}
