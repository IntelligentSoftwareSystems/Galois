; ModuleID = 'static-ctor.bc'
@llvm.global_ctors = appending global [1 x { i32, void ()* }] [ { i32, void ()* } { i32 0, void ()* @DMP_init } ]		; <[1 x { i32, void ()* }]*> [#uses=0]

declare void @DMP_init()
