// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// adaptspec
Rcpp::List adaptspec(unsigned int nLoop, unsigned int nWarmUp, Rcpp::NumericMatrix xR, Rcpp::List missingIndicesR, Rcpp::List priorList, Rcpp::List tuningList, Rcpp::List startR, Rcpp::List thin, bool showProgress);
RcppExport SEXP _BayesSpec_adaptspec(SEXP nLoopSEXP, SEXP nWarmUpSEXP, SEXP xRSEXP, SEXP missingIndicesRSEXP, SEXP priorListSEXP, SEXP tuningListSEXP, SEXP startRSEXP, SEXP thinSEXP, SEXP showProgressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type nLoop(nLoopSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nWarmUp(nWarmUpSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type missingIndicesR(missingIndicesRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type priorList(priorListSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type tuningList(tuningListSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type startR(startRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< bool >::type showProgress(showProgressSEXP);
    rcpp_result_gen = Rcpp::wrap(adaptspec(nLoop, nWarmUp, xR, missingIndicesR, priorList, tuningList, startR, thin, showProgress));
    return rcpp_result_gen;
END_RCPP
}
// getSampleFilled
Rcpp::List getSampleFilled(Rcpp::NumericMatrix xR, Rcpp::List priorList, Rcpp::List stateList, Rcpp::List tuningList);
RcppExport SEXP _BayesSpec_getSampleFilled(SEXP xRSEXP, SEXP priorListSEXP, SEXP stateListSEXP, SEXP tuningListSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type priorList(priorListSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type stateList(stateListSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type tuningList(tuningListSEXP);
    rcpp_result_gen = Rcpp::wrap(getSampleFilled(xR, priorList, stateList, tuningList));
    return rcpp_result_gen;
END_RCPP
}
// getMetropolisLogRatio
double getMetropolisLogRatio(Rcpp::List currentR, Rcpp::List proposalR, Rcpp::NumericMatrix xR, Rcpp::List priorList, Rcpp::List tuningList);
RcppExport SEXP _BayesSpec_getMetropolisLogRatio(SEXP currentRSEXP, SEXP proposalRSEXP, SEXP xRSEXP, SEXP priorListSEXP, SEXP tuningListSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type currentR(currentRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type proposalR(proposalRSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type priorList(priorListSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type tuningList(tuningListSEXP);
    rcpp_result_gen = Rcpp::wrap(getMetropolisLogRatio(currentR, proposalR, xR, priorList, tuningList));
    return rcpp_result_gen;
END_RCPP
}
// dirichletMixture
Rcpp::List dirichletMixture(unsigned int nLoop, unsigned int nWarmUp, Rcpp::NumericMatrix xR, Rcpp::List missingIndicesR, Rcpp::List priorsR, double alphaPriorShape, double alphaPriorRate, Rcpp::List componentTuningR, bool firstCategoryFixed, Rcpp::List startR, Rcpp::List thin, bool showProgress);
RcppExport SEXP _BayesSpec_dirichletMixture(SEXP nLoopSEXP, SEXP nWarmUpSEXP, SEXP xRSEXP, SEXP missingIndicesRSEXP, SEXP priorsRSEXP, SEXP alphaPriorShapeSEXP, SEXP alphaPriorRateSEXP, SEXP componentTuningRSEXP, SEXP firstCategoryFixedSEXP, SEXP startRSEXP, SEXP thinSEXP, SEXP showProgressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type nLoop(nLoopSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nWarmUp(nWarmUpSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type missingIndicesR(missingIndicesRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type priorsR(priorsRSEXP);
    Rcpp::traits::input_parameter< double >::type alphaPriorShape(alphaPriorShapeSEXP);
    Rcpp::traits::input_parameter< double >::type alphaPriorRate(alphaPriorRateSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type componentTuningR(componentTuningRSEXP);
    Rcpp::traits::input_parameter< bool >::type firstCategoryFixed(firstCategoryFixedSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type startR(startRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< bool >::type showProgress(showProgressSEXP);
    rcpp_result_gen = Rcpp::wrap(dirichletMixture(nLoop, nWarmUp, xR, missingIndicesR, priorsR, alphaPriorShape, alphaPriorRate, componentTuningR, firstCategoryFixed, startR, thin, showProgress));
    return rcpp_result_gen;
END_RCPP
}
// fftForwardR2C
Rcpp::ComplexVector fftForwardR2C(Rcpp::NumericVector inputR);
RcppExport SEXP _BayesSpec_fftForwardR2C(SEXP inputRSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type inputR(inputRSEXP);
    rcpp_result_gen = Rcpp::wrap(fftForwardR2C(inputR));
    return rcpp_result_gen;
END_RCPP
}
// fftForwardC2C
Rcpp::ComplexVector fftForwardC2C(Rcpp::ComplexVector inputR);
RcppExport SEXP _BayesSpec_fftForwardC2C(SEXP inputRSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::ComplexVector >::type inputR(inputRSEXP);
    rcpp_result_gen = Rcpp::wrap(fftForwardC2C(inputR));
    return rcpp_result_gen;
END_RCPP
}
// independentMixture
Rcpp::List independentMixture(unsigned int nLoop, unsigned int nWarmUp, Rcpp::NumericMatrix xR, Rcpp::List missingIndicesR, Rcpp::List priorsR, Rcpp::NumericVector weightsPriorR, Rcpp::List componentTuningR, bool firstCategoryFixed, Rcpp::List startR, Rcpp::List thin, bool showProgress);
RcppExport SEXP _BayesSpec_independentMixture(SEXP nLoopSEXP, SEXP nWarmUpSEXP, SEXP xRSEXP, SEXP missingIndicesRSEXP, SEXP priorsRSEXP, SEXP weightsPriorRSEXP, SEXP componentTuningRSEXP, SEXP firstCategoryFixedSEXP, SEXP startRSEXP, SEXP thinSEXP, SEXP showProgressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type nLoop(nLoopSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nWarmUp(nWarmUpSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type missingIndicesR(missingIndicesRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type priorsR(priorsRSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type weightsPriorR(weightsPriorRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type componentTuningR(componentTuningRSEXP);
    Rcpp::traits::input_parameter< bool >::type firstCategoryFixed(firstCategoryFixedSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type startR(startRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< bool >::type showProgress(showProgressSEXP);
    rcpp_result_gen = Rcpp::wrap(independentMixture(nLoop, nWarmUp, xR, missingIndicesR, priorsR, weightsPriorR, componentTuningR, firstCategoryFixed, startR, thin, showProgress));
    return rcpp_result_gen;
END_RCPP
}
// logisticStickBreakingMixture
Rcpp::List logisticStickBreakingMixture(unsigned int nLoop, unsigned int nWarmUp, Rcpp::NumericMatrix xR, Rcpp::List missingIndicesR, Rcpp::NumericMatrix designMatrixR, Rcpp::List priorsR, Rcpp::NumericMatrix priorMeanR, Rcpp::NumericMatrix priorPrecisionR, double tauPriorASquared, double tauPriorNu, Rcpp::List componentTuningR, bool firstCategoryFixed, unsigned int nSplineBases, Rcpp::List startR, Rcpp::List thin, bool showProgress, bool mpi);
RcppExport SEXP _BayesSpec_logisticStickBreakingMixture(SEXP nLoopSEXP, SEXP nWarmUpSEXP, SEXP xRSEXP, SEXP missingIndicesRSEXP, SEXP designMatrixRSEXP, SEXP priorsRSEXP, SEXP priorMeanRSEXP, SEXP priorPrecisionRSEXP, SEXP tauPriorASquaredSEXP, SEXP tauPriorNuSEXP, SEXP componentTuningRSEXP, SEXP firstCategoryFixedSEXP, SEXP nSplineBasesSEXP, SEXP startRSEXP, SEXP thinSEXP, SEXP showProgressSEXP, SEXP mpiSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type nLoop(nLoopSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nWarmUp(nWarmUpSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type missingIndicesR(missingIndicesRSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type designMatrixR(designMatrixRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type priorsR(priorsRSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type priorMeanR(priorMeanRSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type priorPrecisionR(priorPrecisionRSEXP);
    Rcpp::traits::input_parameter< double >::type tauPriorASquared(tauPriorASquaredSEXP);
    Rcpp::traits::input_parameter< double >::type tauPriorNu(tauPriorNuSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type componentTuningR(componentTuningRSEXP);
    Rcpp::traits::input_parameter< bool >::type firstCategoryFixed(firstCategoryFixedSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nSplineBases(nSplineBasesSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type startR(startRSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< bool >::type showProgress(showProgressSEXP);
    Rcpp::traits::input_parameter< bool >::type mpi(mpiSEXP);
    rcpp_result_gen = Rcpp::wrap(logisticStickBreakingMixture(nLoop, nWarmUp, xR, missingIndicesR, designMatrixR, priorsR, priorMeanR, priorPrecisionR, tauPriorASquared, tauPriorNu, componentTuningR, firstCategoryFixed, nSplineBases, startR, thin, showProgress, mpi));
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
// timeVaryingSpectraSamples
NumericVector timeVaryingSpectraSamples(IntegerVector nSegments, IntegerMatrix cutPoints, NumericVector beta, unsigned int nFrequencies, unsigned int timeStep);
RcppExport SEXP _BayesSpec_timeVaryingSpectraSamples(SEXP nSegmentsSEXP, SEXP cutPointsSEXP, SEXP betaSEXP, SEXP nFrequenciesSEXP, SEXP timeStepSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< IntegerVector >::type nSegments(nSegmentsSEXP);
    Rcpp::traits::input_parameter< IntegerMatrix >::type cutPoints(cutPointsSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nFrequencies(nFrequenciesSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type timeStep(timeStepSEXP);
    rcpp_result_gen = Rcpp::wrap(timeVaryingSpectraSamples(nSegments, cutPoints, beta, nFrequencies, timeStep));
    return rcpp_result_gen;
END_RCPP
}
// timeVaryingSpectraMixtureMeanCategories
NumericVector timeVaryingSpectraMixtureMeanCategories(NumericVector componentSamples, IntegerMatrix categories);
RcppExport SEXP _BayesSpec_timeVaryingSpectraMixtureMeanCategories(SEXP componentSamplesSEXP, SEXP categoriesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type componentSamples(componentSamplesSEXP);
    Rcpp::traits::input_parameter< IntegerMatrix >::type categories(categoriesSEXP);
    rcpp_result_gen = Rcpp::wrap(timeVaryingSpectraMixtureMeanCategories(componentSamples, categories));
    return rcpp_result_gen;
END_RCPP
}
// timeVaryingSpectraMixtureMeanProbabilities
NumericVector timeVaryingSpectraMixtureMeanProbabilities(NumericVector componentSamples, NumericVector probabilities);
RcppExport SEXP _BayesSpec_timeVaryingSpectraMixtureMeanProbabilities(SEXP componentSamplesSEXP, SEXP probabilitiesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type componentSamples(componentSamplesSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type probabilities(probabilitiesSEXP);
    rcpp_result_gen = Rcpp::wrap(timeVaryingSpectraMixtureMeanProbabilities(componentSamples, probabilities));
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
// sampleWhittleMissingR
Rcpp::NumericVector sampleWhittleMissingR(Rcpp::NumericVector xR, Rcpp::IntegerVector missingIndicesR, Rcpp::NumericVector halfSpectrumR);
RcppExport SEXP _BayesSpec_sampleWhittleMissingR(SEXP xRSEXP, SEXP missingIndicesRSEXP, SEXP halfSpectrumRSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type xR(xRSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type missingIndicesR(missingIndicesRSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type halfSpectrumR(halfSpectrumRSEXP);
    rcpp_result_gen = Rcpp::wrap(sampleWhittleMissingR(xR, missingIndicesR, halfSpectrumR));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_BayesSpec_adaptspec", (DL_FUNC) &_BayesSpec_adaptspec, 9},
    {"_BayesSpec_getSampleFilled", (DL_FUNC) &_BayesSpec_getSampleFilled, 4},
    {"_BayesSpec_getMetropolisLogRatio", (DL_FUNC) &_BayesSpec_getMetropolisLogRatio, 5},
    {"_BayesSpec_dirichletMixture", (DL_FUNC) &_BayesSpec_dirichletMixture, 12},
    {"_BayesSpec_fftForwardR2C", (DL_FUNC) &_BayesSpec_fftForwardR2C, 1},
    {"_BayesSpec_fftForwardC2C", (DL_FUNC) &_BayesSpec_fftForwardC2C, 1},
    {"_BayesSpec_independentMixture", (DL_FUNC) &_BayesSpec_independentMixture, 11},
    {"_BayesSpec_logisticStickBreakingMixture", (DL_FUNC) &_BayesSpec_logisticStickBreakingMixture, 17},
    {"_BayesSpec_randGamma", (DL_FUNC) &_BayesSpec_randGamma, 3},
    {"_BayesSpec_rlogbeta", (DL_FUNC) &_BayesSpec_rlogbeta, 3},
    {"_BayesSpec_rloggamma", (DL_FUNC) &_BayesSpec_rloggamma, 3},
    {"_BayesSpec_splines_basis1d_demmler_reinsch", (DL_FUNC) &_BayesSpec_splines_basis1d_demmler_reinsch, 2},
    {"_BayesSpec_timeVaryingSpectraSamples", (DL_FUNC) &_BayesSpec_timeVaryingSpectraSamples, 5},
    {"_BayesSpec_timeVaryingSpectraMixtureMeanCategories", (DL_FUNC) &_BayesSpec_timeVaryingSpectraMixtureMeanCategories, 2},
    {"_BayesSpec_timeVaryingSpectraMixtureMeanProbabilities", (DL_FUNC) &_BayesSpec_timeVaryingSpectraMixtureMeanProbabilities, 2},
    {"_BayesSpec_logWhittleLikelihood", (DL_FUNC) &_BayesSpec_logWhittleLikelihood, 3},
    {"_BayesSpec_logWhittleLikelihoodBeta", (DL_FUNC) &_BayesSpec_logWhittleLikelihoodBeta, 4},
    {"_BayesSpec_logWhittleLikelihoodBetaDeriv", (DL_FUNC) &_BayesSpec_logWhittleLikelihoodBetaDeriv, 4},
    {"_BayesSpec_sampleWhittleMissingR", (DL_FUNC) &_BayesSpec_sampleWhittleMissingR, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_BayesSpec(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
