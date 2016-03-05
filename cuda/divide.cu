// dst[i] = a[i] / b[i]
extern "C" __global__ void
divide(float* __restrict__  dst, float* __restrict__  a, float* __restrict__ b, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

	if(i < N) {
		if((a[i] == 0) || (b[i] == 0)) {
			dst[i] = 0.0;
		} else {
			dst[i] = a[i] / b[i];
		}
	}
}
