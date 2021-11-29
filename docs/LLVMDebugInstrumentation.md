# Instrumenting LLVM IR

This document provides a short description about LLVM IR instrumentation techniques. The motivation for this feature is to increase transparency of model execution on a target and ease bug investigation process.


## Overview
There are two types of implementation supported: function body and function call instrumentations.
- Function body instrumentation. One can specify what function bodies he\she wants to trace. It can be achieved by specializing function name or by giving regular expression. Below example shows that every line in the libjit_element_div_kernel_i32 function will be complemented by printouts. I.e. after every line (except alloca and some specific instructions) there will be trace showing what value was written out. The trace consist of instruction number (which is global counter, i.e. if few functions body is instrumented then every new function will have counter continued from previous). On top of that one can specify the printout function that needs to be used to output traces.
If there is a function call in the body then it will be handled in default manner: there will be trace saying that we are about to call function, then all its parameters (integers will be printed always but complex structs only if there is corresponding function with name pretty_print_<struct_name>(struct_name* ) in the context) and exit trace will also be added.

    **-llvm-code-debug-trace-instrumentation="libjit_element_div_kernel_i32:body"**

- Function calls instrumentation. By default it will wrap all function calls that corresponds to given regular expression into traces before and after. Before and after can be either default instrumentation that prints function name and its input parameters or one can specify the functions to use. The specified functions have to have the same signature as original function.

    **-llvm-code-debug-trace-instrumentation="foo:call[:before_foo[:after_foo]]"**

On top of that there is possibility to specify printing function by giving its name. The print out function has to have the same signature as printf

**-llvm-debug-trace-print-function-name=my_printf**

There is option to output instrumented IR by providing in command line (works only with debug glow build):

**-debug-glow-only=debug-instrumentation**

## Examples

- Body instrumentation example
image-classifier --backend=CPU -expected-labels=0 -image-mode=0to1 -model-input-name=data_0 **-llvm-code-debug-trace-instrumentation=libjit_element_div_kernel_i32:body -llvm-debug-trace-print-function-name=printf -debug-glow-only=debug-instrumentation** -m=mnist.onnx 0_1009.png

Before:
```
; Function Attrs: nounwind optnone uwtable
define internal i32 @libjit_element_div_kernel_i32(i64, i32*, i32*, i32*) #0 {
  %5 = alloca i64, align 8
  %6 = alloca i32*, align 8
  %7 = alloca i32*, align 8
  %8 = alloca i32*, align 8
  store i64 %0, i64* %5, align 8
  store i32* %1, i32** %6, align 8
  store i32* %2, i32** %7, align 8
  store i32* %3, i32** %8, align 8
  %9 = load i32*, i32** %6, align 8
  %10 = load i64, i64* %5, align 8
  %11 = getelementptr inbounds i32, i32* %9, i64 %10
  %12 = load i32, i32* %11, align 4
  %13 = load i32*, i32** %7, align 8
  %14 = load i64, i64* %5, align 8
  %15 = getelementptr inbounds i32, i32* %13, i64 %14
  %16 = load i32, i32* %15, align 4
  %17 = sdiv i32 %12, %16
  ret i32 %17
}
```


After:
```
@.str.9 = private unnamed_addr constant [40 x i8] c"Instruction number %u ,stored value %d\0A\00", align 1
@.str.12 = private unnamed_addr constant [40 x i8] c"Instruction number %u ,stored value %p\0A\00", align 1
@.str.14 = private unnamed_addr constant [20 x i8] c"Function called %s\0A\00", align 1
@.str.16 = private unnamed_addr constant [20 x i8] c"Function exited %s\0A\00", align 1

; Function Attrs: nounwind optnone uwtable
define internal i32 @libjit_element_div_kernel_i32(i64, i32*, i32*, i32*) #0 {
  %5 = alloca i64, align 8
  %6 = alloca i32*, align 8
  %7 = alloca i32*, align 8
  %8 = alloca i32*, align 8
  store i64 %0, i64* %5, align 8
  %9 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.12, i32 0, i32 0), i64 0, i64* %5)
  store i32* %1, i32** %6, align 8
  %10 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.12, i32 0, i32 0), i64 1, i32** %6)
  store i32* %2, i32** %7, align 8
  %11 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.12, i32 0, i32 0), i64 2, i32** %7)
  store i32* %3, i32** %8, align 8
  %12 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.12, i32 0, i32 0), i64 3, i32** %8)
  %13 = load i32*, i32** %6, align 8
  %14 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.9, i32 0, i32 0), i64 4, i32* %13)
  %15 = load i64, i64* %5, align 8
  %16 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.9, i32 0, i32 0), i64 5, i64 %15)
  %17 = getelementptr inbounds i32, i32* %13, i64 %15
  %18 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.9, i32 0, i32 0), i64 6, i32* %17)
  %19 = load i32, i32* %17, align 4
  %20 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.9, i32 0, i32 0), i64 7, i32 %19)
  %21 = load i32*, i32** %7, align 8
  %22 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.9, i32 0, i32 0), i64 8, i32* %21)
  %23 = load i64, i64* %5, align 8
  %24 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.9, i32 0, i32 0), i64 9, i64 %23)
  %25 = getelementptr inbounds i32, i32* %21, i64 %23
  %26 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.9, i32 0, i32 0), i64 10, i32* %25)
  %27 = load i32, i32* %25, align 4
  %28 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.9, i32 0, i32 0), i64 11, i32 %27)
  %29 = sdiv i32 %19, %27
  %30 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.9, i32 0, i32 0), i64 12, i32 %29)
  ret i32 %29
}
```

- Call example
image-classifier --backend=CPU -expected-labels=0 -image-mode=0to1 -model-input-name=data_0 **-llvm-code-debug-trace-instrumentation=libjit_element_relu_f:call -llvm-debug-trace-print-function-name=printf -debug-glow-only=debug-instrumentation** -m=mnist.onnx 0_1009.png

Before:
```
  %1 = phi i64 [ 0, %entry ], [ %nextvar, %loop ]
  %2 = call float @libjit_element_relu_f(i64 %1, float* %0)
  %buffer.element.addr = getelementptr float, float* %0, i64 %1
```

After:
```
@.str.9 = private unnamed_addr constant [40 x i8] c"Instruction number %u ,stored value %d\0A\00", align 1
@.str.12 = private unnamed_addr constant [40 x i8] c"Instruction number %u ,stored value %p\0A\00", align 1
@.str.14 = private unnamed_addr constant [20 x i8] c"Function called %s\0A\00", align 1
@.str.16 = private unnamed_addr constant [20 x i8] c"Function exited %s\0A\00", align 1
@.str.18 = private unnamed_addr constant [22 x i8] c"libjit_element_relu_f\00", align 1
@.str.20 = private unnamed_addr constant [10 x i8] c"\09arg: %d\0A\00", align 1
@.str.37 = private unnamed_addr constant [11 x i8] c"\09arg: ...\0A\00", align 1
@.str.39 = private unnamed_addr constant [22 x i8] c"libjit_element_relu_f\00", align 1
@.str.44 = private unnamed_addr constant [10 x i8] c"\09arg: %d\0A\00", align 1
@.str.45 = private unnamed_addr constant [11 x i8] c"\09arg: ...\0A\00", align 1

%1 = phi i64 [ 0, %entry ], [ %nextvar, %loop ]
%2 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.14, i32 0, i32 0), i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.18, i32 0, i32 0))
%3 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.20, i32 0, i32 0), i64 %1)
%4 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.37, i32 0, i32 0))
%5 = call float @libjit_element_relu_f(i64 %1, float* %0)
%6 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.16, i32 0, i32 0), i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.18, i32 0, i32 0))
%buffer.element.addr = getelementptr float, float* %0, i64 %1
```

## Further improvements
1. Provide possibility to consume lib or LLVM bitcode file that will provide pretty_print_* functions
2. Improve stability, i.e. customize body printouts based on LLVM instruction result type
3. Add printout of debug information: file names and line numbers. Or llvm IR line mapping that can be used by gdb at live time debugging.
