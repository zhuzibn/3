package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

var (
	Msat        ScalarParam // Saturation magnetization in A/m
	M_full      setter      // non-reduced magnetization in A/m
	B_demag     setter      // demag field in Tesla
	E_demag     = NewGetScalar("E_demag", "J", "Magnetostatic energy", getDemagEnergy)
	EnableDemag = true // enable/disable demag field
	bsat        param
	demagconv_  *cuda.DemagConvolution // does the heavy lifting and provides FFTM
)

func init() {
	Msat.init("Msat", "A/m", "Saturation magnetization")

	M_full.init(3, &globalmesh, "m_full", "A/m", "Unnormalized magnetization", func(dst *data.Slice) {
		msat, r := Msat.Get()
		if r {
			defer cuda.RecycleBuffer(msat)
		}
		for c := 0; c < 3; c++ {
			cuda.Mul(dst.Comp(c), M.buffer.Comp(c), msat)
		}
	})

	DeclVar("EnableDemag", &EnableDemag, "Enables/disables demag (default=true)")

	bsat.init_(1, "Bsat", "T", func() {
		panic("todo")
	})

	B_demag.init(3, &globalmesh, "B_demag", "T", "Magnetostatic field (T)", func(b *data.Slice) {
		if EnableDemag {
			demagConv().Exec(b, M.buffer, bsat.Gpu1(), regions.Gpu())
		} else {
			cuda.Zero(b)
		}
	})

	registerEnergy(getDemagEnergy)
}

func demagConv() *cuda.DemagConvolution {
	if demagconv_ == nil {
		demagconv_ = cuda.NewDemag(Mesh())
	}
	return demagconv_
}

// Returns the current demag energy in Joules.
func getDemagEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_demag)
}

func safediv(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}
