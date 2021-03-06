package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
	"sync"
	"unsafe"
)

// CUDA handle for reducemaxabs kernel
var reducemaxabs_code cu.Function

// Stores the arguments for reducemaxabs kernel invocation
type reducemaxabs_args_t struct {
	arg_src     unsafe.Pointer
	arg_dst     unsafe.Pointer
	arg_initVal float32
	arg_n       int
	argptr      [4]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for reducemaxabs kernel invocation
var reducemaxabs_args reducemaxabs_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	reducemaxabs_args.argptr[0] = unsafe.Pointer(&reducemaxabs_args.arg_src)
	reducemaxabs_args.argptr[1] = unsafe.Pointer(&reducemaxabs_args.arg_dst)
	reducemaxabs_args.argptr[2] = unsafe.Pointer(&reducemaxabs_args.arg_initVal)
	reducemaxabs_args.argptr[3] = unsafe.Pointer(&reducemaxabs_args.arg_n)
}

// Wrapper for reducemaxabs CUDA kernel, asynchronous.
func k_reducemaxabs_async(src unsafe.Pointer, dst unsafe.Pointer, initVal float32, n int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("reducemaxabs")
	}

	reducemaxabs_args.Lock()
	defer reducemaxabs_args.Unlock()

	if reducemaxabs_code == 0 {
		reducemaxabs_code = fatbinLoad(reducemaxabs_map, "reducemaxabs")
	}

	reducemaxabs_args.arg_src = src
	reducemaxabs_args.arg_dst = dst
	reducemaxabs_args.arg_initVal = initVal
	reducemaxabs_args.arg_n = n

	args := reducemaxabs_args.argptr[:]
	cu.LaunchKernel(reducemaxabs_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("reducemaxabs")
	}
}

// maps compute capability on PTX code for reducemaxabs kernel.
var reducemaxabs_map = map[int]string{0: "",
	20: reducemaxabs_ptx_20,
	30: reducemaxabs_ptx_30,
	35: reducemaxabs_ptx_35,
	50: reducemaxabs_ptx_50,
	52: reducemaxabs_ptx_52,
	53: reducemaxabs_ptx_53,
	60: reducemaxabs_ptx_60,
	61: reducemaxabs_ptx_61,
	62: reducemaxabs_ptx_62,
	70: reducemaxabs_ptx_70}

// reducemaxabs PTX code for various compute capabilities.
const (
	reducemaxabs_ptx_20 = `
.version 3.2
.target sm_20
.address_size 64

.global .align 1 .b8 $str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90, 0};

.visible .entry reducemaxabs(
	.param .u64 reducemaxabs_param_0,
	.param .u64 reducemaxabs_param_1,
	.param .f32 reducemaxabs_param_2,
	.param .u32 reducemaxabs_param_3
)
{
	.reg .pred 	%p<8>;
	.reg .s32 	%r<17>;
	.reg .f32 	%f<31>;
	.reg .s64 	%rd<13>;
	// demoted variable
	.shared .align 4 .b8 reducemaxabs$__cuda_local_var_33861_35_non_const_sdata[2048];

	ld.param.u64 	%rd4, [reducemaxabs_param_0];
	ld.param.u64 	%rd5, [reducemaxabs_param_1];
	ld.param.f32 	%f30, [reducemaxabs_param_2];
	ld.param.u32 	%r9, [reducemaxabs_param_3];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	.loc 1 8 1
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r15, %r16, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r16, %r11;
	.loc 1 8 1
	setp.ge.s32	%p1, %r15, %r9;
	@%p1 bra 	BB0_2;

BB0_1:
	.loc 1 8 1
	mul.wide.s32 	%rd6, %r15, 4;
	add.s64 	%rd7, %rd2, %rd6;
	ld.global.f32 	%f5, [%rd7];
	.loc 2 2750 10
	abs.f32 	%f6, %f5;
	.loc 2 2770 10
	max.f32 	%f30, %f30, %f6;
	.loc 1 8 1
	add.s32 	%r15, %r15, %r4;
	.loc 1 8 1
	setp.lt.s32	%p2, %r15, %r9;
	@%p2 bra 	BB0_1;

BB0_2:
	.loc 1 8 1
	mul.wide.s32 	%rd8, %r2, 4;
	mov.u64 	%rd9, reducemaxabs$__cuda_local_var_33861_35_non_const_sdata;
	add.s64 	%rd3, %rd9, %rd8;
	st.shared.f32 	[%rd3], %f30;
	bar.sync 	0;
	.loc 1 8 1
	setp.lt.u32	%p3, %r16, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	.loc 1 8 1
	mov.u32 	%r7, %r16;
	shr.u32 	%r16, %r7, 1;
	.loc 1 8 1
	setp.ge.u32	%p4, %r2, %r16;
	@%p4 bra 	BB0_5;

	.loc 1 8 1
	ld.shared.f32 	%f7, [%rd3];
	add.s32 	%r12, %r16, %r2;
	mul.wide.u32 	%rd10, %r12, 4;
	add.s64 	%rd12, %rd9, %rd10;
	ld.shared.f32 	%f8, [%rd12];
	.loc 2 2770 10
	max.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%rd3], %f9;

BB0_5:
	.loc 1 8 1
	bar.sync 	0;
	.loc 1 8 1
	setp.gt.u32	%p5, %r7, 131;
	@%p5 bra 	BB0_3;

BB0_6:
	.loc 1 8 1
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	.loc 1 8 1
	ld.volatile.shared.f32 	%f10, [%rd3];
	ld.volatile.shared.f32 	%f11, [%rd3+128];
	.loc 2 2770 10
	max.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%rd3], %f12;
	.loc 1 8 1
	ld.volatile.shared.f32 	%f13, [%rd3+64];
	ld.volatile.shared.f32 	%f14, [%rd3];
	.loc 2 2770 10
	max.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%rd3], %f15;
	.loc 1 8 1
	ld.volatile.shared.f32 	%f16, [%rd3+32];
	ld.volatile.shared.f32 	%f17, [%rd3];
	.loc 2 2770 10
	max.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%rd3], %f18;
	.loc 1 8 1
	ld.volatile.shared.f32 	%f19, [%rd3+16];
	ld.volatile.shared.f32 	%f20, [%rd3];
	.loc 2 2770 10
	max.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%rd3], %f21;
	.loc 1 8 1
	ld.volatile.shared.f32 	%f22, [%rd3+8];
	ld.volatile.shared.f32 	%f23, [%rd3];
	.loc 2 2770 10
	max.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%rd3], %f24;
	.loc 1 8 1
	ld.volatile.shared.f32 	%f25, [%rd3+4];
	ld.volatile.shared.f32 	%f26, [%rd3];
	.loc 2 2770 10
	max.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%rd3], %f27;

BB0_8:
	.loc 1 8 1
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	.loc 1 8 1
	ld.shared.f32 	%f28, [reducemaxabs$__cuda_local_var_33861_35_non_const_sdata];
	.loc 2 2750 10
	abs.f32 	%f29, %f28;
	.loc 1 8 37
	mov.b32 	 %r13, %f29;
	.loc 2 3781 3
	atom.global.max.s32 	%r14, [%rd1], %r13;

BB0_10:
	.loc 1 9 2
	ret;
}


`
	reducemaxabs_ptx_30 = `
.version 4.0
.target sm_30
.address_size 64


.visible .entry reducemaxabs(
	.param .u64 reducemaxabs_param_0,
	.param .u64 reducemaxabs_param_1,
	.param .f32 reducemaxabs_param_2,
	.param .u32 reducemaxabs_param_3
)
{
	.reg .pred 	%p<8>;
	.reg .s32 	%r<17>;
	.reg .f32 	%f<31>;
	.reg .s64 	%rd<13>;
	// demoted variable
	.shared .align 4 .b8 reducemaxabs$__cuda_local_var_34206_35_non_const_sdata[2048];

	ld.param.u64 	%rd4, [reducemaxabs_param_0];
	ld.param.u64 	%rd3, [reducemaxabs_param_1];
	ld.param.f32 	%f30, [reducemaxabs_param_2];
	ld.param.u32 	%r9, [reducemaxabs_param_3];
	cvta.to.global.u64 	%rd1, %rd4;
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r15, %r16, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r11, %r16;
	setp.ge.s32	%p1, %r15, %r9;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd5, %r15, 4;
	add.s64 	%rd6, %rd1, %rd5;
	ld.global.f32 	%f5, [%rd6];
	abs.f32 	%f6, %f5;
	max.f32 	%f30, %f30, %f6;
	add.s32 	%r15, %r15, %r4;
	setp.lt.s32	%p2, %r15, %r9;
	@%p2 bra 	BB0_1;

BB0_2:
	mul.wide.s32 	%rd7, %r2, 4;
	mov.u64 	%rd8, reducemaxabs$__cuda_local_var_34206_35_non_const_sdata;
	add.s64 	%rd2, %rd8, %rd7;
	st.shared.f32 	[%rd2], %f30;
	bar.sync 	0;
	setp.lt.u32	%p3, %r16, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	mov.u32 	%r7, %r16;
	shr.u32 	%r16, %r7, 1;
	setp.ge.u32	%p4, %r2, %r16;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%rd2];
	add.s32 	%r12, %r16, %r2;
	mul.wide.u32 	%rd9, %r12, 4;
	add.s64 	%rd11, %rd8, %rd9;
	ld.shared.f32 	%f8, [%rd11];
	max.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%rd2], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r7, 131;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%rd2];
	ld.volatile.shared.f32 	%f11, [%rd2+128];
	max.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%rd2], %f12;
	ld.volatile.shared.f32 	%f13, [%rd2+64];
	ld.volatile.shared.f32 	%f14, [%rd2];
	max.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%rd2], %f15;
	ld.volatile.shared.f32 	%f16, [%rd2+32];
	ld.volatile.shared.f32 	%f17, [%rd2];
	max.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%rd2], %f18;
	ld.volatile.shared.f32 	%f19, [%rd2+16];
	ld.volatile.shared.f32 	%f20, [%rd2];
	max.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%rd2], %f21;
	ld.volatile.shared.f32 	%f22, [%rd2+8];
	ld.volatile.shared.f32 	%f23, [%rd2];
	max.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%rd2], %f24;
	ld.volatile.shared.f32 	%f25, [%rd2+4];
	ld.volatile.shared.f32 	%f26, [%rd2];
	max.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%rd2], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	cvta.to.global.u64 	%rd12, %rd3;
	ld.shared.f32 	%f28, [reducemaxabs$__cuda_local_var_34206_35_non_const_sdata];
	abs.f32 	%f29, %f28;
	mov.b32 	 %r13, %f29;
	atom.global.max.s32 	%r14, [%rd12], %r13;

BB0_10:
	ret;
}


`
	reducemaxabs_ptx_35 = `
.version 4.1
.target sm_35
.address_size 64


.weak .func  (.param .b32 func_retval0) cudaMalloc(
	.param .b64 cudaMalloc_param_0,
	.param .b64 cudaMalloc_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaFuncGetAttributes(
	.param .b64 cudaFuncGetAttributes_param_0,
	.param .b64 cudaFuncGetAttributes_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaDeviceGetAttribute(
	.param .b64 cudaDeviceGetAttribute_param_0,
	.param .b32 cudaDeviceGetAttribute_param_1,
	.param .b32 cudaDeviceGetAttribute_param_2
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaGetDevice(
	.param .b64 cudaGetDevice_param_0
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessor(
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_0,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_1,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_2,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_3
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.visible .entry reducemaxabs(
	.param .u64 reducemaxabs_param_0,
	.param .u64 reducemaxabs_param_1,
	.param .f32 reducemaxabs_param_2,
	.param .u32 reducemaxabs_param_3
)
{
	.reg .pred 	%p<8>;
	.reg .s32 	%r<17>;
	.reg .f32 	%f<31>;
	.reg .s64 	%rd<13>;
	// demoted variable
	.shared .align 4 .b8 reducemaxabs$__cuda_local_var_34786_35_non_const_sdata[2048];

	ld.param.u64 	%rd4, [reducemaxabs_param_0];
	ld.param.u64 	%rd3, [reducemaxabs_param_1];
	ld.param.f32 	%f30, [reducemaxabs_param_2];
	ld.param.u32 	%r9, [reducemaxabs_param_3];
	cvta.to.global.u64 	%rd1, %rd4;
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r15, %r16, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r11, %r16;
	setp.ge.s32	%p1, %r15, %r9;
	@%p1 bra 	BB5_2;

BB5_1:
	mul.wide.s32 	%rd5, %r15, 4;
	add.s64 	%rd6, %rd1, %rd5;
	ld.global.nc.f32 	%f5, [%rd6];
	abs.f32 	%f6, %f5;
	max.f32 	%f30, %f30, %f6;
	add.s32 	%r15, %r15, %r4;
	setp.lt.s32	%p2, %r15, %r9;
	@%p2 bra 	BB5_1;

BB5_2:
	mul.wide.s32 	%rd7, %r2, 4;
	mov.u64 	%rd8, reducemaxabs$__cuda_local_var_34786_35_non_const_sdata;
	add.s64 	%rd2, %rd8, %rd7;
	st.shared.f32 	[%rd2], %f30;
	bar.sync 	0;
	setp.lt.u32	%p3, %r16, 66;
	@%p3 bra 	BB5_6;

BB5_3:
	mov.u32 	%r7, %r16;
	shr.u32 	%r16, %r7, 1;
	setp.ge.u32	%p4, %r2, %r16;
	@%p4 bra 	BB5_5;

	ld.shared.f32 	%f7, [%rd2];
	add.s32 	%r12, %r16, %r2;
	mul.wide.u32 	%rd9, %r12, 4;
	add.s64 	%rd11, %rd8, %rd9;
	ld.shared.f32 	%f8, [%rd11];
	max.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%rd2], %f9;

BB5_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r7, 131;
	@%p5 bra 	BB5_3;

BB5_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB5_8;

	ld.volatile.shared.f32 	%f10, [%rd2];
	ld.volatile.shared.f32 	%f11, [%rd2+128];
	max.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%rd2], %f12;
	ld.volatile.shared.f32 	%f13, [%rd2+64];
	ld.volatile.shared.f32 	%f14, [%rd2];
	max.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%rd2], %f15;
	ld.volatile.shared.f32 	%f16, [%rd2+32];
	ld.volatile.shared.f32 	%f17, [%rd2];
	max.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%rd2], %f18;
	ld.volatile.shared.f32 	%f19, [%rd2+16];
	ld.volatile.shared.f32 	%f20, [%rd2];
	max.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%rd2], %f21;
	ld.volatile.shared.f32 	%f22, [%rd2+8];
	ld.volatile.shared.f32 	%f23, [%rd2];
	max.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%rd2], %f24;
	ld.volatile.shared.f32 	%f25, [%rd2+4];
	ld.volatile.shared.f32 	%f26, [%rd2];
	max.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%rd2], %f27;

BB5_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB5_10;

	cvta.to.global.u64 	%rd12, %rd3;
	ld.shared.f32 	%f28, [reducemaxabs$__cuda_local_var_34786_35_non_const_sdata];
	abs.f32 	%f29, %f28;
	mov.b32 	 %r13, %f29;
	atom.global.max.s32 	%r14, [%rd12], %r13;

BB5_10:
	ret;
}


`
	reducemaxabs_ptx_50 = `
.version 4.3
.target sm_50
.address_size 64

	// .weak	cudaMalloc

.weak .func  (.param .b32 func_retval0) cudaMalloc(
	.param .b64 cudaMalloc_param_0,
	.param .b64 cudaMalloc_param_1
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaFuncGetAttributes
.weak .func  (.param .b32 func_retval0) cudaFuncGetAttributes(
	.param .b64 cudaFuncGetAttributes_param_0,
	.param .b64 cudaFuncGetAttributes_param_1
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaDeviceGetAttribute
.weak .func  (.param .b32 func_retval0) cudaDeviceGetAttribute(
	.param .b64 cudaDeviceGetAttribute_param_0,
	.param .b32 cudaDeviceGetAttribute_param_1,
	.param .b32 cudaDeviceGetAttribute_param_2
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaGetDevice
.weak .func  (.param .b32 func_retval0) cudaGetDevice(
	.param .b64 cudaGetDevice_param_0
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaOccupancyMaxActiveBlocksPerMultiprocessor
.weak .func  (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessor(
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_0,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_1,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_2,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_3
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
.weak .func  (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_0,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_1,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_2,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_3,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_4
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .globl	reducemaxabs
.visible .entry reducemaxabs(
	.param .u64 reducemaxabs_param_0,
	.param .u64 reducemaxabs_param_1,
	.param .f32 reducemaxabs_param_2,
	.param .u32 reducemaxabs_param_3
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<31>;
	.reg .b32 	%r<17>;
	.reg .b64 	%rd<13>;
	// demoted variable
	.shared .align 4 .b8 reducemaxabs$__cuda_local_var_42648_35_non_const_sdata[2048];

	ld.param.u64 	%rd4, [reducemaxabs_param_0];
	ld.param.u64 	%rd3, [reducemaxabs_param_1];
	ld.param.f32 	%f30, [reducemaxabs_param_2];
	ld.param.u32 	%r9, [reducemaxabs_param_3];
	cvta.to.global.u64 	%rd1, %rd4;
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r15, %r16, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r11, %r16;
	setp.ge.s32	%p1, %r15, %r9;
	@%p1 bra 	BB6_2;

BB6_1:
	mul.wide.s32 	%rd5, %r15, 4;
	add.s64 	%rd6, %rd1, %rd5;
	ld.global.nc.f32 	%f5, [%rd6];
	abs.f32 	%f6, %f5;
	max.f32 	%f30, %f30, %f6;
	add.s32 	%r15, %r15, %r4;
	setp.lt.s32	%p2, %r15, %r9;
	@%p2 bra 	BB6_1;

BB6_2:
	mul.wide.s32 	%rd7, %r2, 4;
	mov.u64 	%rd8, reducemaxabs$__cuda_local_var_42648_35_non_const_sdata;
	add.s64 	%rd2, %rd8, %rd7;
	st.shared.f32 	[%rd2], %f30;
	bar.sync 	0;
	setp.lt.u32	%p3, %r16, 66;
	@%p3 bra 	BB6_6;

BB6_3:
	mov.u32 	%r7, %r16;
	shr.u32 	%r16, %r7, 1;
	setp.ge.u32	%p4, %r2, %r16;
	@%p4 bra 	BB6_5;

	ld.shared.f32 	%f7, [%rd2];
	add.s32 	%r12, %r16, %r2;
	mul.wide.u32 	%rd9, %r12, 4;
	add.s64 	%rd11, %rd8, %rd9;
	ld.shared.f32 	%f8, [%rd11];
	max.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%rd2], %f9;

BB6_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r7, 131;
	@%p5 bra 	BB6_3;

BB6_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB6_8;

	ld.volatile.shared.f32 	%f10, [%rd2];
	ld.volatile.shared.f32 	%f11, [%rd2+128];
	max.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%rd2], %f12;
	ld.volatile.shared.f32 	%f13, [%rd2+64];
	ld.volatile.shared.f32 	%f14, [%rd2];
	max.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%rd2], %f15;
	ld.volatile.shared.f32 	%f16, [%rd2+32];
	ld.volatile.shared.f32 	%f17, [%rd2];
	max.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%rd2], %f18;
	ld.volatile.shared.f32 	%f19, [%rd2+16];
	ld.volatile.shared.f32 	%f20, [%rd2];
	max.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%rd2], %f21;
	ld.volatile.shared.f32 	%f22, [%rd2+8];
	ld.volatile.shared.f32 	%f23, [%rd2];
	max.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%rd2], %f24;
	ld.volatile.shared.f32 	%f25, [%rd2+4];
	ld.volatile.shared.f32 	%f26, [%rd2];
	max.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%rd2], %f27;

BB6_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB6_10;

	ld.shared.f32 	%f28, [reducemaxabs$__cuda_local_var_42648_35_non_const_sdata];
	abs.f32 	%f29, %f28;
	mov.b32 	 %r13, %f29;
	cvta.to.global.u64 	%rd12, %rd3;
	atom.global.max.s32 	%r14, [%rd12], %r13;

BB6_10:
	ret;
}


`
	reducemaxabs_ptx_52 = `
.version 4.3
.target sm_52
.address_size 64

	// .weak	cudaMalloc

.weak .func  (.param .b32 func_retval0) cudaMalloc(
	.param .b64 cudaMalloc_param_0,
	.param .b64 cudaMalloc_param_1
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaFuncGetAttributes
.weak .func  (.param .b32 func_retval0) cudaFuncGetAttributes(
	.param .b64 cudaFuncGetAttributes_param_0,
	.param .b64 cudaFuncGetAttributes_param_1
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaDeviceGetAttribute
.weak .func  (.param .b32 func_retval0) cudaDeviceGetAttribute(
	.param .b64 cudaDeviceGetAttribute_param_0,
	.param .b32 cudaDeviceGetAttribute_param_1,
	.param .b32 cudaDeviceGetAttribute_param_2
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaGetDevice
.weak .func  (.param .b32 func_retval0) cudaGetDevice(
	.param .b64 cudaGetDevice_param_0
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaOccupancyMaxActiveBlocksPerMultiprocessor
.weak .func  (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessor(
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_0,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_1,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_2,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_3
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
.weak .func  (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_0,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_1,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_2,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_3,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_4
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .globl	reducemaxabs
.visible .entry reducemaxabs(
	.param .u64 reducemaxabs_param_0,
	.param .u64 reducemaxabs_param_1,
	.param .f32 reducemaxabs_param_2,
	.param .u32 reducemaxabs_param_3
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<31>;
	.reg .b32 	%r<17>;
	.reg .b64 	%rd<13>;
	// demoted variable
	.shared .align 4 .b8 reducemaxabs$__cuda_local_var_42648_35_non_const_sdata[2048];

	ld.param.u64 	%rd4, [reducemaxabs_param_0];
	ld.param.u64 	%rd3, [reducemaxabs_param_1];
	ld.param.f32 	%f30, [reducemaxabs_param_2];
	ld.param.u32 	%r9, [reducemaxabs_param_3];
	cvta.to.global.u64 	%rd1, %rd4;
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r15, %r16, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r11, %r16;
	setp.ge.s32	%p1, %r15, %r9;
	@%p1 bra 	BB6_2;

BB6_1:
	mul.wide.s32 	%rd5, %r15, 4;
	add.s64 	%rd6, %rd1, %rd5;
	ld.global.nc.f32 	%f5, [%rd6];
	abs.f32 	%f6, %f5;
	max.f32 	%f30, %f30, %f6;
	add.s32 	%r15, %r15, %r4;
	setp.lt.s32	%p2, %r15, %r9;
	@%p2 bra 	BB6_1;

BB6_2:
	mul.wide.s32 	%rd7, %r2, 4;
	mov.u64 	%rd8, reducemaxabs$__cuda_local_var_42648_35_non_const_sdata;
	add.s64 	%rd2, %rd8, %rd7;
	st.shared.f32 	[%rd2], %f30;
	bar.sync 	0;
	setp.lt.u32	%p3, %r16, 66;
	@%p3 bra 	BB6_6;

BB6_3:
	mov.u32 	%r7, %r16;
	shr.u32 	%r16, %r7, 1;
	setp.ge.u32	%p4, %r2, %r16;
	@%p4 bra 	BB6_5;

	ld.shared.f32 	%f7, [%rd2];
	add.s32 	%r12, %r16, %r2;
	mul.wide.u32 	%rd9, %r12, 4;
	add.s64 	%rd11, %rd8, %rd9;
	ld.shared.f32 	%f8, [%rd11];
	max.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%rd2], %f9;

BB6_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r7, 131;
	@%p5 bra 	BB6_3;

BB6_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB6_8;

	ld.volatile.shared.f32 	%f10, [%rd2];
	ld.volatile.shared.f32 	%f11, [%rd2+128];
	max.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%rd2], %f12;
	ld.volatile.shared.f32 	%f13, [%rd2+64];
	ld.volatile.shared.f32 	%f14, [%rd2];
	max.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%rd2], %f15;
	ld.volatile.shared.f32 	%f16, [%rd2+32];
	ld.volatile.shared.f32 	%f17, [%rd2];
	max.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%rd2], %f18;
	ld.volatile.shared.f32 	%f19, [%rd2+16];
	ld.volatile.shared.f32 	%f20, [%rd2];
	max.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%rd2], %f21;
	ld.volatile.shared.f32 	%f22, [%rd2+8];
	ld.volatile.shared.f32 	%f23, [%rd2];
	max.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%rd2], %f24;
	ld.volatile.shared.f32 	%f25, [%rd2+4];
	ld.volatile.shared.f32 	%f26, [%rd2];
	max.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%rd2], %f27;

BB6_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB6_10;

	ld.shared.f32 	%f28, [reducemaxabs$__cuda_local_var_42648_35_non_const_sdata];
	abs.f32 	%f29, %f28;
	mov.b32 	 %r13, %f29;
	cvta.to.global.u64 	%rd12, %rd3;
	atom.global.max.s32 	%r14, [%rd12], %r13;

BB6_10:
	ret;
}


`
	reducemaxabs_ptx_53 = `
.version 4.3
.target sm_53
.address_size 64

	// .weak	cudaMalloc

.weak .func  (.param .b32 func_retval0) cudaMalloc(
	.param .b64 cudaMalloc_param_0,
	.param .b64 cudaMalloc_param_1
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaFuncGetAttributes
.weak .func  (.param .b32 func_retval0) cudaFuncGetAttributes(
	.param .b64 cudaFuncGetAttributes_param_0,
	.param .b64 cudaFuncGetAttributes_param_1
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaDeviceGetAttribute
.weak .func  (.param .b32 func_retval0) cudaDeviceGetAttribute(
	.param .b64 cudaDeviceGetAttribute_param_0,
	.param .b32 cudaDeviceGetAttribute_param_1,
	.param .b32 cudaDeviceGetAttribute_param_2
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaGetDevice
.weak .func  (.param .b32 func_retval0) cudaGetDevice(
	.param .b64 cudaGetDevice_param_0
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaOccupancyMaxActiveBlocksPerMultiprocessor
.weak .func  (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessor(
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_0,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_1,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_2,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_3
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
.weak .func  (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_0,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_1,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_2,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_3,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_4
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .globl	reducemaxabs
.visible .entry reducemaxabs(
	.param .u64 reducemaxabs_param_0,
	.param .u64 reducemaxabs_param_1,
	.param .f32 reducemaxabs_param_2,
	.param .u32 reducemaxabs_param_3
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<31>;
	.reg .b32 	%r<17>;
	.reg .b64 	%rd<13>;
	// demoted variable
	.shared .align 4 .b8 reducemaxabs$__cuda_local_var_42648_35_non_const_sdata[2048];

	ld.param.u64 	%rd4, [reducemaxabs_param_0];
	ld.param.u64 	%rd3, [reducemaxabs_param_1];
	ld.param.f32 	%f30, [reducemaxabs_param_2];
	ld.param.u32 	%r9, [reducemaxabs_param_3];
	cvta.to.global.u64 	%rd1, %rd4;
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r15, %r16, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r11, %r16;
	setp.ge.s32	%p1, %r15, %r9;
	@%p1 bra 	BB6_2;

BB6_1:
	mul.wide.s32 	%rd5, %r15, 4;
	add.s64 	%rd6, %rd1, %rd5;
	ld.global.nc.f32 	%f5, [%rd6];
	abs.f32 	%f6, %f5;
	max.f32 	%f30, %f30, %f6;
	add.s32 	%r15, %r15, %r4;
	setp.lt.s32	%p2, %r15, %r9;
	@%p2 bra 	BB6_1;

BB6_2:
	mul.wide.s32 	%rd7, %r2, 4;
	mov.u64 	%rd8, reducemaxabs$__cuda_local_var_42648_35_non_const_sdata;
	add.s64 	%rd2, %rd8, %rd7;
	st.shared.f32 	[%rd2], %f30;
	bar.sync 	0;
	setp.lt.u32	%p3, %r16, 66;
	@%p3 bra 	BB6_6;

BB6_3:
	mov.u32 	%r7, %r16;
	shr.u32 	%r16, %r7, 1;
	setp.ge.u32	%p4, %r2, %r16;
	@%p4 bra 	BB6_5;

	ld.shared.f32 	%f7, [%rd2];
	add.s32 	%r12, %r16, %r2;
	mul.wide.u32 	%rd9, %r12, 4;
	add.s64 	%rd11, %rd8, %rd9;
	ld.shared.f32 	%f8, [%rd11];
	max.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%rd2], %f9;

BB6_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r7, 131;
	@%p5 bra 	BB6_3;

BB6_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB6_8;

	ld.volatile.shared.f32 	%f10, [%rd2];
	ld.volatile.shared.f32 	%f11, [%rd2+128];
	max.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%rd2], %f12;
	ld.volatile.shared.f32 	%f13, [%rd2+64];
	ld.volatile.shared.f32 	%f14, [%rd2];
	max.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%rd2], %f15;
	ld.volatile.shared.f32 	%f16, [%rd2+32];
	ld.volatile.shared.f32 	%f17, [%rd2];
	max.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%rd2], %f18;
	ld.volatile.shared.f32 	%f19, [%rd2+16];
	ld.volatile.shared.f32 	%f20, [%rd2];
	max.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%rd2], %f21;
	ld.volatile.shared.f32 	%f22, [%rd2+8];
	ld.volatile.shared.f32 	%f23, [%rd2];
	max.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%rd2], %f24;
	ld.volatile.shared.f32 	%f25, [%rd2+4];
	ld.volatile.shared.f32 	%f26, [%rd2];
	max.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%rd2], %f27;

BB6_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB6_10;

	ld.shared.f32 	%f28, [reducemaxabs$__cuda_local_var_42648_35_non_const_sdata];
	abs.f32 	%f29, %f28;
	mov.b32 	 %r13, %f29;
	cvta.to.global.u64 	%rd12, %rd3;
	atom.global.max.s32 	%r14, [%rd12], %r13;

BB6_10:
	ret;
}


`
	reducemaxabs_ptx_60 = `
.version 5.0
.target sm_60
.address_size 64

	// .globl	reducemaxabs

.visible .entry reducemaxabs(
	.param .u64 reducemaxabs_param_0,
	.param .u64 reducemaxabs_param_1,
	.param .f32 reducemaxabs_param_2,
	.param .u32 reducemaxabs_param_3
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<31>;
	.reg .b32 	%r<17>;
	.reg .b64 	%rd<13>;
	// demoted variable
	.shared .align 4 .b8 reducemaxabs$__cuda_local_var_16860_35_non_const_sdata[2048];

	ld.param.u64 	%rd4, [reducemaxabs_param_0];
	ld.param.u64 	%rd3, [reducemaxabs_param_1];
	ld.param.f32 	%f30, [reducemaxabs_param_2];
	ld.param.u32 	%r9, [reducemaxabs_param_3];
	cvta.to.global.u64 	%rd1, %rd4;
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r15, %r16, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r11, %r16;
	setp.ge.s32	%p1, %r15, %r9;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd5, %r15, 4;
	add.s64 	%rd6, %rd1, %rd5;
	ld.global.nc.f32 	%f5, [%rd6];
	abs.f32 	%f6, %f5;
	max.f32 	%f30, %f30, %f6;
	add.s32 	%r15, %r15, %r4;
	setp.lt.s32	%p2, %r15, %r9;
	@%p2 bra 	BB0_1;

BB0_2:
	mul.wide.s32 	%rd7, %r2, 4;
	mov.u64 	%rd8, reducemaxabs$__cuda_local_var_16860_35_non_const_sdata;
	add.s64 	%rd2, %rd8, %rd7;
	st.shared.f32 	[%rd2], %f30;
	bar.sync 	0;
	setp.lt.u32	%p3, %r16, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	mov.u32 	%r7, %r16;
	shr.u32 	%r16, %r7, 1;
	setp.ge.u32	%p4, %r2, %r16;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%rd2];
	add.s32 	%r12, %r16, %r2;
	mul.wide.u32 	%rd9, %r12, 4;
	add.s64 	%rd11, %rd8, %rd9;
	ld.shared.f32 	%f8, [%rd11];
	max.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%rd2], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r7, 131;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%rd2];
	ld.volatile.shared.f32 	%f11, [%rd2+128];
	max.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%rd2], %f12;
	ld.volatile.shared.f32 	%f13, [%rd2+64];
	ld.volatile.shared.f32 	%f14, [%rd2];
	max.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%rd2], %f15;
	ld.volatile.shared.f32 	%f16, [%rd2+32];
	ld.volatile.shared.f32 	%f17, [%rd2];
	max.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%rd2], %f18;
	ld.volatile.shared.f32 	%f19, [%rd2+16];
	ld.volatile.shared.f32 	%f20, [%rd2];
	max.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%rd2], %f21;
	ld.volatile.shared.f32 	%f22, [%rd2+8];
	ld.volatile.shared.f32 	%f23, [%rd2];
	max.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%rd2], %f24;
	ld.volatile.shared.f32 	%f25, [%rd2+4];
	ld.volatile.shared.f32 	%f26, [%rd2];
	max.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%rd2], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f28, [reducemaxabs$__cuda_local_var_16860_35_non_const_sdata];
	abs.f32 	%f29, %f28;
	mov.b32 	 %r13, %f29;
	cvta.to.global.u64 	%rd12, %rd3;
	atom.global.max.s32 	%r14, [%rd12], %r13;

BB0_10:
	ret;
}


`
	reducemaxabs_ptx_61 = `
.version 5.0
.target sm_61
.address_size 64

	// .globl	reducemaxabs

.visible .entry reducemaxabs(
	.param .u64 reducemaxabs_param_0,
	.param .u64 reducemaxabs_param_1,
	.param .f32 reducemaxabs_param_2,
	.param .u32 reducemaxabs_param_3
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<31>;
	.reg .b32 	%r<17>;
	.reg .b64 	%rd<13>;
	// demoted variable
	.shared .align 4 .b8 reducemaxabs$__cuda_local_var_16860_35_non_const_sdata[2048];

	ld.param.u64 	%rd4, [reducemaxabs_param_0];
	ld.param.u64 	%rd3, [reducemaxabs_param_1];
	ld.param.f32 	%f30, [reducemaxabs_param_2];
	ld.param.u32 	%r9, [reducemaxabs_param_3];
	cvta.to.global.u64 	%rd1, %rd4;
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r15, %r16, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r11, %r16;
	setp.ge.s32	%p1, %r15, %r9;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd5, %r15, 4;
	add.s64 	%rd6, %rd1, %rd5;
	ld.global.nc.f32 	%f5, [%rd6];
	abs.f32 	%f6, %f5;
	max.f32 	%f30, %f30, %f6;
	add.s32 	%r15, %r15, %r4;
	setp.lt.s32	%p2, %r15, %r9;
	@%p2 bra 	BB0_1;

BB0_2:
	mul.wide.s32 	%rd7, %r2, 4;
	mov.u64 	%rd8, reducemaxabs$__cuda_local_var_16860_35_non_const_sdata;
	add.s64 	%rd2, %rd8, %rd7;
	st.shared.f32 	[%rd2], %f30;
	bar.sync 	0;
	setp.lt.u32	%p3, %r16, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	mov.u32 	%r7, %r16;
	shr.u32 	%r16, %r7, 1;
	setp.ge.u32	%p4, %r2, %r16;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%rd2];
	add.s32 	%r12, %r16, %r2;
	mul.wide.u32 	%rd9, %r12, 4;
	add.s64 	%rd11, %rd8, %rd9;
	ld.shared.f32 	%f8, [%rd11];
	max.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%rd2], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r7, 131;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%rd2];
	ld.volatile.shared.f32 	%f11, [%rd2+128];
	max.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%rd2], %f12;
	ld.volatile.shared.f32 	%f13, [%rd2+64];
	ld.volatile.shared.f32 	%f14, [%rd2];
	max.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%rd2], %f15;
	ld.volatile.shared.f32 	%f16, [%rd2+32];
	ld.volatile.shared.f32 	%f17, [%rd2];
	max.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%rd2], %f18;
	ld.volatile.shared.f32 	%f19, [%rd2+16];
	ld.volatile.shared.f32 	%f20, [%rd2];
	max.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%rd2], %f21;
	ld.volatile.shared.f32 	%f22, [%rd2+8];
	ld.volatile.shared.f32 	%f23, [%rd2];
	max.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%rd2], %f24;
	ld.volatile.shared.f32 	%f25, [%rd2+4];
	ld.volatile.shared.f32 	%f26, [%rd2];
	max.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%rd2], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f28, [reducemaxabs$__cuda_local_var_16860_35_non_const_sdata];
	abs.f32 	%f29, %f28;
	mov.b32 	 %r13, %f29;
	cvta.to.global.u64 	%rd12, %rd3;
	atom.global.max.s32 	%r14, [%rd12], %r13;

BB0_10:
	ret;
}


`
	reducemaxabs_ptx_62 = `
.version 5.0
.target sm_62
.address_size 64

	// .globl	reducemaxabs

.visible .entry reducemaxabs(
	.param .u64 reducemaxabs_param_0,
	.param .u64 reducemaxabs_param_1,
	.param .f32 reducemaxabs_param_2,
	.param .u32 reducemaxabs_param_3
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<31>;
	.reg .b32 	%r<17>;
	.reg .b64 	%rd<13>;
	// demoted variable
	.shared .align 4 .b8 reducemaxabs$__cuda_local_var_16860_35_non_const_sdata[2048];

	ld.param.u64 	%rd4, [reducemaxabs_param_0];
	ld.param.u64 	%rd3, [reducemaxabs_param_1];
	ld.param.f32 	%f30, [reducemaxabs_param_2];
	ld.param.u32 	%r9, [reducemaxabs_param_3];
	cvta.to.global.u64 	%rd1, %rd4;
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r15, %r16, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r11, %r16;
	setp.ge.s32	%p1, %r15, %r9;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd5, %r15, 4;
	add.s64 	%rd6, %rd1, %rd5;
	ld.global.nc.f32 	%f5, [%rd6];
	abs.f32 	%f6, %f5;
	max.f32 	%f30, %f30, %f6;
	add.s32 	%r15, %r15, %r4;
	setp.lt.s32	%p2, %r15, %r9;
	@%p2 bra 	BB0_1;

BB0_2:
	mul.wide.s32 	%rd7, %r2, 4;
	mov.u64 	%rd8, reducemaxabs$__cuda_local_var_16860_35_non_const_sdata;
	add.s64 	%rd2, %rd8, %rd7;
	st.shared.f32 	[%rd2], %f30;
	bar.sync 	0;
	setp.lt.u32	%p3, %r16, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	mov.u32 	%r7, %r16;
	shr.u32 	%r16, %r7, 1;
	setp.ge.u32	%p4, %r2, %r16;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%rd2];
	add.s32 	%r12, %r16, %r2;
	mul.wide.u32 	%rd9, %r12, 4;
	add.s64 	%rd11, %rd8, %rd9;
	ld.shared.f32 	%f8, [%rd11];
	max.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%rd2], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r7, 131;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%rd2];
	ld.volatile.shared.f32 	%f11, [%rd2+128];
	max.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%rd2], %f12;
	ld.volatile.shared.f32 	%f13, [%rd2+64];
	ld.volatile.shared.f32 	%f14, [%rd2];
	max.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%rd2], %f15;
	ld.volatile.shared.f32 	%f16, [%rd2+32];
	ld.volatile.shared.f32 	%f17, [%rd2];
	max.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%rd2], %f18;
	ld.volatile.shared.f32 	%f19, [%rd2+16];
	ld.volatile.shared.f32 	%f20, [%rd2];
	max.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%rd2], %f21;
	ld.volatile.shared.f32 	%f22, [%rd2+8];
	ld.volatile.shared.f32 	%f23, [%rd2];
	max.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%rd2], %f24;
	ld.volatile.shared.f32 	%f25, [%rd2+4];
	ld.volatile.shared.f32 	%f26, [%rd2];
	max.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%rd2], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f28, [reducemaxabs$__cuda_local_var_16860_35_non_const_sdata];
	abs.f32 	%f29, %f28;
	mov.b32 	 %r13, %f29;
	cvta.to.global.u64 	%rd12, %rd3;
	atom.global.max.s32 	%r14, [%rd12], %r13;

BB0_10:
	ret;
}


`
	reducemaxabs_ptx_70 = `
.version 6.0
.target sm_70
.address_size 64

	// .globl	reducemaxabs

.visible .entry reducemaxabs(
	.param .u64 reducemaxabs_param_0,
	.param .u64 reducemaxabs_param_1,
	.param .f32 reducemaxabs_param_2,
	.param .u32 reducemaxabs_param_3
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<32>;
	.reg .b32 	%r<23>;
	.reg .b64 	%rd<7>;
	// demoted variable
	.shared .align 4 .b8 reducemaxabs$__cuda_local_var_15592_35_non_const_sdata[2048];

	ld.param.u64 	%rd3, [reducemaxabs_param_0];
	ld.param.u64 	%rd2, [reducemaxabs_param_1];
	ld.param.f32 	%f31, [reducemaxabs_param_2];
	ld.param.u32 	%r10, [reducemaxabs_param_3];
	cvta.to.global.u64 	%rd1, %rd3;
	mov.u32 	%r22, %ntid.x;
	mov.u32 	%r11, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r21, %r22, %r11, %r2;
	mov.u32 	%r12, %nctaid.x;
	mul.lo.s32 	%r4, %r12, %r22;
	setp.ge.s32	%p1, %r21, %r10;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd4, %r21, 4;
	add.s64 	%rd5, %rd1, %rd4;
	ld.global.nc.f32 	%f5, [%rd5];
	abs.f32 	%f6, %f5;
	max.f32 	%f31, %f31, %f6;
	add.s32 	%r21, %r21, %r4;
	setp.lt.s32	%p2, %r21, %r10;
	@%p2 bra 	BB0_1;

BB0_2:
	shl.b32 	%r13, %r2, 2;
	mov.u32 	%r14, reducemaxabs$__cuda_local_var_15592_35_non_const_sdata;
	add.s32 	%r7, %r14, %r13;
	st.shared.f32 	[%r7], %f31;
	bar.sync 	0;
	setp.lt.u32	%p3, %r22, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	shr.u32 	%r9, %r22, 1;
	setp.ge.u32	%p4, %r2, %r9;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%r7];
	add.s32 	%r15, %r9, %r2;
	shl.b32 	%r16, %r15, 2;
	add.s32 	%r18, %r14, %r16;
	ld.shared.f32 	%f8, [%r18];
	max.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%r7], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r22, 131;
	mov.u32 	%r22, %r9;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%r7];
	ld.volatile.shared.f32 	%f11, [%r7+128];
	max.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%r7], %f12;
	ld.volatile.shared.f32 	%f13, [%r7+64];
	ld.volatile.shared.f32 	%f14, [%r7];
	max.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%r7], %f15;
	ld.volatile.shared.f32 	%f16, [%r7+32];
	ld.volatile.shared.f32 	%f17, [%r7];
	max.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%r7], %f18;
	ld.volatile.shared.f32 	%f19, [%r7+16];
	ld.volatile.shared.f32 	%f20, [%r7];
	max.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%r7], %f21;
	ld.volatile.shared.f32 	%f22, [%r7+8];
	ld.volatile.shared.f32 	%f23, [%r7];
	max.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%r7], %f24;
	ld.volatile.shared.f32 	%f25, [%r7+4];
	ld.volatile.shared.f32 	%f26, [%r7];
	max.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%r7], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f28, [reducemaxabs$__cuda_local_var_15592_35_non_const_sdata];
	abs.f32 	%f29, %f28;
	mov.b32 	 %r19, %f29;
	cvta.to.global.u64 	%rd6, %rd2;
	atom.global.max.s32 	%r20, [%rd6], %r19;

BB0_10:
	ret;
}


`
)
