shtools_wrapper.so: shtools_wrapper.f90
	gfortran shtools_wrapper.f90 -I$(SHTOOLSMODPATH) -L$(SHTOOLSLIBPATH) -lSHTOOLS -lfftw3 -lm -llapack -lblas -O3 -m64 -fPIC -shared -o shtools_wrapper.so

