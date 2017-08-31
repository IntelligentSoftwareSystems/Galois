################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Cholesky.cpp \
../main.cpp 

CXX_SRCS += \
../DoubleArgFunction.cxx \
../Element.cxx \
../MatrixGenerator.cxx \
../Tier.cxx 

OBJS += \
./Cholesky.o \
./DoubleArgFunction.o \
./Element.o \
./MatrixGenerator.o \
./Tier.o \
./main.o 

CPP_DEPS += \
./Cholesky.d \
./main.d 

CXX_DEPS += \
./DoubleArgFunction.d \
./Element.d \
./MatrixGenerator.d \
./Tier.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cxx
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


