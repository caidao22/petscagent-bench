
CFLAGS	 =
FFLAGS	 =
CPPFLAGS =
FPPFLAGS =
LOCDIR	 =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

generated_code: generated_code.o
	-${CLINKER} -o generated_code generated_code.o ${PETSC_TS_LIB}
	${RM} generated_code.o

