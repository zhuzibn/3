// Original implementation by Mykola Dvornik for mumax2
// Modified for mumax3 by Arne Vansteenkiste, 2013

#include <stdint.h>
#include "float3.h"
#include "constants.h"

extern "C" __global__ void
addoommfslonczewskitorque(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
                     float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz, float* __restrict__ jz,
		     float* __restrict__ px, float* __restrict__ py, float* __restrict__ pz,
                     float* __restrict__ msatLUT, float* __restrict__ alphaLUT, float flt,
                     float* __restrict__ pfixLUT, float* __restrict__ pfreeLUT,
                     float* __restrict__ lambdafixLUT, float* __restrict__ lambdafreeLUT,
		     float* __restrict__ epsilonPrimeLUT,
                     uint8_t* __restrict__ regions, int N) {

	int I =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (I < N) {

		float3 m = make_float3(mx[I], my[I], mz[I]);
		float  J = jz[I];
		float3 p = normalized(make_float3(px[I], py[I], pz[I]));

		// read parameters
		uint8_t region       = regions[I];

		float  Ms           = msatLUT[region];
		float  alpha        = alphaLUT[region];
		float  pfix         = pfixLUT[region];
		float  pfree        = pfreeLUT[region];
		float  lambdafix    = lambdafixLUT[region];
		float  lambdafree   = lambdafreeLUT[region];
		float  epsilonPrime = epsilonPrimeLUT[region];

		if (J == 0.0f || Ms == 0.0f) {
			return;
		}

		float beta    = (HBAR / QE) * (J / (2.0f *flt*Ms) );
		float lambdafix2 = lambdafix * lambdafix;
		float lambdafree2 = lambdafree * lambdafree;
		float lambdafreePlus = sqrt(lambdafree2 + 1.0f);
		float lambdafixPlus = sqrt(lambdafix2 + 1.0f);
		float lambdafreeMinus = sqrt(lambdafree2 - 1.0f);
		float lambdafixMinus = sqrt(lambdafix2 - 1.0f);
		float plus_ratio = lambdafreePlus / lambdafixPlus;
		float minus_ratio = 1.0f;
		if (lambdafreeMinus > 0) {
		   	minus_ratio = lambdafixMinus / lambdafreeMinus;
		}
		// Compute q_plus and q_minus
		float plus_factor = pfix * lambdafix2 * plus_ratio;
		float minus_factor = pfree * lambdafree2 * minus_ratio;
		float q_plus = plus_factor + minus_factor;
		float q_minus = plus_factor - minus_factor;
		float lplus2 = lambdafreePlus * lambdafixPlus;
		float lminus2 = lambdafreeMinus * lambdafixMinus;
		float pdotm = dot(p, m);
		float A_plus = lplus2 + (lminus2 * pdotm);
		float A_minus = lplus2 - (lminus2 * pdotm);
		float epsilon = (q_plus / A_plus) - (q_minus / A_minus);

		float A = beta * epsilon;
		float B = beta * epsilonPrime;

		float gilb     = 1.0f / (1.0f + alpha * alpha);
		float mxpxmFac = gilb * (A - alpha * B);
		float pxmFac   = gilb * (B - alpha * A);

		float3 pxm      = cross(p, m);
		float3 mxpxm    = cross(m, pxm);

		tx[I] += mxpxmFac * mxpxm.x + pxmFac * pxm.x;
		ty[I] += mxpxmFac * mxpxm.y + pxmFac * pxm.y;
		tz[I] += mxpxmFac * mxpxm.z + pxmFac * pxm.z;
	}
}
