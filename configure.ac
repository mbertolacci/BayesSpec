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

AC_ARG_ENABLE(fftw3, [--enable-fftw Use fftw3 if it's available])

if test "$enable_fftw3" = "yes"; then
  PKG_CXXFLAGS="-DEIGEN_FFTW_DEFAULT ${PKG_CXXFLAGS}"
  LIBS="-lfftw3 ${LIBS}"
fi

PKG_CXXFLAGS="-DEIGEN_DONT_PARALLELIZE -I spectra/include ${PKG_CXXFLAGS}"

AC_SUBST(LIBS)
AC_SUBST(PKG_CXXFLAGS)

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
