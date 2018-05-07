// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// adaptspec
Rcpp::List adaptspec(unsigned int nLoop, unsigned int nWarmUp, Rcpp::NumericMatrix xR, Rcpp::List priorList, double probMM1, double varInflate, double burnInVarInflate, unsigned int nSegmentsStart, bool showProgress);
RcppExport SEXP _BayesSpec_adaptspec(SEXP nLoopSEXP, SEXP nWarmUpSEXP, SEXP xRSEXP, SEXP priorListSEXP, SEXP probMM1SEXP, SEXP varInflateSEXP, SEXP burnInVarInflateSEXP, SEXP nSegmentsStartSEXP, SEXP showProgressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type nLoop(nLoopSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nWarmUp(nWarmUpSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type priorList(priorListSEXP);
    Rcpp::traits::input_parameter< double >::type probMM1(probMM1SEXP);
    Rcpp::traits::input_parameter< double >::type varInflate(varInflateSEXP);
    Rcpp::traits::input_parameter< double >::type burnInVarInflate(burnInVarInflateSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nSegmentsStart(nSegmentsStartSEXP);
    Rcpp::traits::input_parameter< bool >::type showProgress(showProgressSEXP);
    rcpp_result_gen = Rcpp::wrap(adaptspec(nLoop, nWarmUp, xR, priorList, probMM1, varInflate, burnInVarInflate, nSegmentsStart, showProgress));
    return rcpp_result_gen;
END_RCPP
}
// getSampleDefault
Rcpp::List getSampleDefault(Rcpp::NumericMatrix xR, Rcpp::List priorList, unsigned int nStartingSegments);
RcppExport SEXP _BayesSpec_getSampleDefault(SEXP xRSEXP, SEXP priorListSEXP, SEXP nStartingSegmentsSEXP) {
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
Rcpp::List getSampleFilled(Rcpp::NumericMatrix xR, Rcpp::List priorList, Rcpp::List stateList);
RcppExport SEXP _BayesSpec_getSampleFilled(SEXP xRSEXP, SEXP priorListSEXP, SEXP stateListSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type priorList(priorListSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type stateList(stateListSEXP);
    rcpp_result_gen = Rcpp::wrap(getSampleFilled(xR, priorList, stateList));
    return rcpp_result_gen;
END_RCPP
}
// getMetropolisLogRatio
double getMetropolisLogRatio(Rcpp::List currentR, Rcpp::List proposalR, Rcpp::NumericMatrix xR, Rcpp::List priorList);
RcppExport SEXP _BayesSpec_getMetropolisLogRatio(SEXP currentRSEXP, SEXP proposalRSEXP, SEXP xRSEXP, SEXP priorListSEXP) {
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
// dirichletMixture
Rcpp::List dirichletMixture(unsigned int nLoop, unsigned int nWarmUp, Rcpp::NumericMatrix xR, Rcpp::List priorsR, double alphaPriorShape, double alphaPriorRate, Rcpp::IntegerVector initialCategoriesR, double probMM1, double varInflate, double burnInVarInflate, bool firstCategoryFixed, bool showProgress);
RcppExport SEXP _BayesSpec_dirichletMixture(SEXP nLoopSEXP, SEXP nWarmUpSEXP, SEXP xRSEXP, SEXP priorsRSEXP, SEXP alphaPriorShapeSEXP, SEXP alphaPriorRateSEXP, SEXP initialCategoriesRSEXP, SEXP probMM1SEXP, SEXP varInflateSEXP, SEXP burnInVarInflateSEXP, SEXP firstCategoryFixedSEXP, SEXP showProgressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type nLoop(nLoopSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nWarmUp(nWarmUpSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type priorsR(priorsRSEXP);
    Rcpp::traits::input_parameter< double >::type alphaPriorShape(alphaPriorShapeSEXP);
    Rcpp::traits::input_parameter< double >::type alphaPriorRate(alphaPriorRateSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type initialCategoriesR(initialCategoriesRSEXP);
    Rcpp::traits::input_parameter< double >::type probMM1(probMM1SEXP);
    Rcpp::traits::input_parameter< double >::type varInflate(varInflateSEXP);
    Rcpp::traits::input_parameter< double >::type burnInVarInflate(burnInVarInflateSEXP);
    Rcpp::traits::input_parameter< bool >::type firstCategoryFixed(firstCategoryFixedSEXP);
    Rcpp::traits::input_parameter< bool >::type showProgress(showProgressSEXP);
    rcpp_result_gen = Rcpp::wrap(dirichletMixture(nLoop, nWarmUp, xR, priorsR, alphaPriorShape, alphaPriorRate, initialCategoriesR, probMM1, varInflate, burnInVarInflate, firstCategoryFixed, showProgress));
    return rcpp_result_gen;
END_RCPP
}
// independentMixture
Rcpp::List independentMixture(unsigned int nLoop, unsigned int nWarmUp, Rcpp::NumericMatrix xR, Rcpp::List priorsR, Rcpp::NumericVector weightsPriorR, Rcpp::IntegerVector initialCategoriesR, double probMM1, double varInflate, double burnInVarInflate, bool firstCategoryFixed, bool showProgress);
RcppExport SEXP _BayesSpec_independentMixture(SEXP nLoopSEXP, SEXP nWarmUpSEXP, SEXP xRSEXP, SEXP priorsRSEXP, SEXP weightsPriorRSEXP, SEXP initialCategoriesRSEXP, SEXP probMM1SEXP, SEXP varInflateSEXP, SEXP burnInVarInflateSEXP, SEXP firstCategoryFixedSEXP, SEXP showProgressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type nLoop(nLoopSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nWarmUp(nWarmUpSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type priorsR(priorsRSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type weightsPriorR(weightsPriorRSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type initialCategoriesR(initialCategoriesRSEXP);
    Rcpp::traits::input_parameter< double >::type probMM1(probMM1SEXP);
    Rcpp::traits::input_parameter< double >::type varInflate(varInflateSEXP);
    Rcpp::traits::input_parameter< double >::type burnInVarInflate(burnInVarInflateSEXP);
    Rcpp::traits::input_parameter< bool >::type firstCategoryFixed(firstCategoryFixedSEXP);
    Rcpp::traits::input_parameter< bool >::type showProgress(showProgressSEXP);
    rcpp_result_gen = Rcpp::wrap(independentMixture(nLoop, nWarmUp, xR, priorsR, weightsPriorR, initialCategoriesR, probMM1, varInflate, burnInVarInflate, firstCategoryFixed, showProgress));
    return rcpp_result_gen;
END_RCPP
}
// randGamma
Rcpp::NumericVector randGamma(int n, double shape, double scale);
RcppExport SEXP _BayesSpec_randGamma(SEXP nSEXP, SEXP shapeSEXP, SEXP scaleSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type shape(shapeSEXP);
    Rcpp::traits::input_parameter< double >::type scale(scaleSEXP);
    rcpp_result_gen = Rcpp::wrap(randGamma(n, shape, scale));
    return rcpp_result_gen;
END_RCPP
}
// rlogbeta
Rcpp::NumericVector rlogbeta(int n, Rcpp::NumericVector a, Rcpp::NumericVector b);
RcppExport SEXP _BayesSpec_rlogbeta(SEXP nSEXP, SEXP aSEXP, SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type a(aSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type b(bSEXP);
    rcpp_result_gen = Rcpp::wrap(rlogbeta(n, a, b));
    return rcpp_result_gen;
END_RCPP
}
// rloggamma
Rcpp::NumericVector rloggamma(int n, Rcpp::NumericVector shape, Rcpp::NumericVector logScale);
RcppExport SEXP _BayesSpec_rloggamma(SEXP nSEXP, SEXP shapeSEXP, SEXP logScaleSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type shape(shapeSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type logScale(logScaleSEXP);
    rcpp_result_gen = Rcpp::wrap(rloggamma(n, shape, logScale));
    return rcpp_result_gen;
END_RCPP
}
// splines_basis1d_demmler_reinsch
Rcpp::NumericMatrix splines_basis1d_demmler_reinsch(Rcpp::NumericVector xR, unsigned int nBases);
RcppExport SEXP _BayesSpec_splines_basis1d_demmler_reinsch(SEXP xRSEXP, SEXP nBasesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nBases(nBasesSEXP);
    rcpp_result_gen = Rcpp::wrap(splines_basis1d_demmler_reinsch(xR, nBases));
    return rcpp_result_gen;
END_RCPP
}
// stickBreakingMixture
Rcpp::List stickBreakingMixture(unsigned int nLoop, unsigned int nWarmUp, Rcpp::NumericMatrix xR, Rcpp::NumericMatrix designMatrixR, Rcpp::List priorsR, Rcpp::NumericMatrix priorMeanR, Rcpp::NumericMatrix priorPrecisionR, double tauPriorASquared, double tauPriorNu, Rcpp::IntegerVector initialCategoriesR, double probMM1, double varInflate, double burnInVarInflate, bool firstCategoryFixed, unsigned int nSplineBases, bool showProgress);
RcppExport SEXP _BayesSpec_stickBreakingMixture(SEXP nLoopSEXP, SEXP nWarmUpSEXP, SEXP xRSEXP, SEXP designMatrixRSEXP, SEXP priorsRSEXP, SEXP priorMeanRSEXP, SEXP priorPrecisionRSEXP, SEXP tauPriorASquaredSEXP, SEXP tauPriorNuSEXP, SEXP initialCategoriesRSEXP, SEXP probMM1SEXP, SEXP varInflateSEXP, SEXP burnInVarInflateSEXP, SEXP firstCategoryFixedSEXP, SEXP nSplineBasesSEXP, SEXP showProgressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type nLoop(nLoopSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nWarmUp(nWarmUpSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type designMatrixR(designMatrixRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type priorsR(priorsRSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type priorMeanR(priorMeanRSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type priorPrecisionR(priorPrecisionRSEXP);
    Rcpp::traits::input_parameter< double >::type tauPriorASquared(tauPriorASquaredSEXP);
    Rcpp::traits::input_parameter< double >::type tauPriorNu(tauPriorNuSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type initialCategoriesR(initialCategoriesRSEXP);
    Rcpp::traits::input_parameter< double >::type probMM1(probMM1SEXP);
    Rcpp::traits::input_parameter< double >::type varInflate(varInflateSEXP);
    Rcpp::traits::input_parameter< double >::type burnInVarInflate(burnInVarInflateSEXP);
    Rcpp::traits::input_parameter< bool >::type firstCategoryFixed(firstCategoryFixedSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nSplineBases(nSplineBasesSEXP);
    Rcpp::traits::input_parameter< bool >::type showProgress(showProgressSEXP);
    rcpp_result_gen = Rcpp::wrap(stickBreakingMixture(nLoop, nWarmUp, xR, designMatrixR, priorsR, priorMeanR, priorPrecisionR, tauPriorASquared, tauPriorNu, initialCategoriesR, probMM1, varInflate, burnInVarInflate, firstCategoryFixed, nSplineBases, showProgress));
    return rcpp_result_gen;
END_RCPP
}
    return rcpp_result_gen;
END_RCPP
}
// logWhittleLikelihood
Rcpp::NumericVector logWhittleLikelihood(Rcpp::NumericVector fHatR, Rcpp::NumericMatrix periodogramR, unsigned int n);
RcppExport SEXP _BayesSpec_logWhittleLikelihood(SEXP fHatRSEXP, SEXP periodogramRSEXP, SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type fHatR(fHatRSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type periodogramR(periodogramRSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(logWhittleLikelihood(fHatR, periodogramR, n));
    return rcpp_result_gen;
END_RCPP
}
// logWhittleLikelihoodBeta
double logWhittleLikelihoodBeta(Rcpp::NumericVector betaR, Rcpp::NumericMatrix nuR, Rcpp::NumericMatrix periodogramR, unsigned int n);
RcppExport SEXP _BayesSpec_logWhittleLikelihoodBeta(SEXP betaRSEXP, SEXP nuRSEXP, SEXP periodogramRSEXP, SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type betaR(betaRSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type nuR(nuRSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type periodogramR(periodogramRSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(logWhittleLikelihoodBeta(betaR, nuR, periodogramR, n));
    return rcpp_result_gen;
END_RCPP
}
// logWhittleLikelihoodBetaDeriv
Rcpp::NumericVector logWhittleLikelihoodBetaDeriv(Rcpp::NumericVector betaR, Rcpp::NumericMatrix nuR, Rcpp::NumericMatrix periodogramR, unsigned int n);
RcppExport SEXP _BayesSpec_logWhittleLikelihoodBetaDeriv(SEXP betaRSEXP, SEXP nuRSEXP, SEXP periodogramRSEXP, SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type betaR(betaRSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type nuR(nuRSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type periodogramR(periodogramRSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(logWhittleLikelihoodBetaDeriv(betaR, nuR, periodogramR, n));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_BayesSpec_adaptspec", (DL_FUNC) &_BayesSpec_adaptspec, 9},
    {"_BayesSpec_getSampleDefault", (DL_FUNC) &_BayesSpec_getSampleDefault, 3},
    {"_BayesSpec_getSampleFilled", (DL_FUNC) &_BayesSpec_getSampleFilled, 3},
    {"_BayesSpec_getMetropolisLogRatio", (DL_FUNC) &_BayesSpec_getMetropolisLogRatio, 4},
    {"_BayesSpec_dirichletMixture", (DL_FUNC) &_BayesSpec_dirichletMixture, 12},
    {"_BayesSpec_independentMixture", (DL_FUNC) &_BayesSpec_independentMixture, 11},
    {"_BayesSpec_randGamma", (DL_FUNC) &_BayesSpec_randGamma, 3},
    {"_BayesSpec_rlogbeta", (DL_FUNC) &_BayesSpec_rlogbeta, 3},
    {"_BayesSpec_rloggamma", (DL_FUNC) &_BayesSpec_rloggamma, 3},
    {"_BayesSpec_splines_basis1d_demmler_reinsch", (DL_FUNC) &_BayesSpec_splines_basis1d_demmler_reinsch, 2},
    {"_BayesSpec_stickBreakingMixture", (DL_FUNC) &_BayesSpec_stickBreakingMixture, 16},
    {"_BayesSpec_logWhittleLikelihood", (DL_FUNC) &_BayesSpec_logWhittleLikelihood, 3},
    {"_BayesSpec_logWhittleLikelihoodBeta", (DL_FUNC) &_BayesSpec_logWhittleLikelihoodBeta, 4},
    {"_BayesSpec_logWhittleLikelihoodBetaDeriv", (DL_FUNC) &_BayesSpec_logWhittleLikelihoodBetaDeriv, 4},
    {NULL, NULL, 0}
};

RcppExport void R_init_BayesSpec(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
