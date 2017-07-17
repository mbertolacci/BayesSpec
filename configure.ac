AC_INIT()

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi

CC=`${R_HOME}/bin/R CMD config CC`
CPP=`${R_HOME}/bin/R CMD config CPP`
CFLAGS=`${R_HOME}/bin/R CMD config CFLAGS`
CXX=`${R_HOME}/bin/R CMD config CXX`
CXXFLAGS=`${R_HOME}/bin/R CMD config CXXFLAGS`

AC_PROG_CC
AC_PROG_CPP

AC_CHECK_LIB(fftw3, fftw_free,
  [
    PKG_CXXFLAGS="-DBAYESSPEC_FFTW ${PKG_CXXFLAGS}"
    LIBS="-lfftw3 ${LIBS}"
  ],
  ,
  [-lm -lfftw3]
)

PKG_CXXFLAGS="-DEIGEN_DONT_PARALLELIZE -I spectra/include ${PKG_CXXFLAGS}"

AC_SUBST(LIBS)
AC_SUBST(PKG_CXXFLAGS)

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
