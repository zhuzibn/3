package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	Alpha                    ScalarParam
	Xi                       ScalarParam
	Pol                      ScalarParam
	Lambda                   ScalarParam
	EpsilonPrime             ScalarParam
	FrozenSpins              ScalarParam
	FixedLayer               VectorParam
	Torque                   vSetter // total torque in T
	LLTorque                 vSetter // Landau-Lifshitz torque/γ0, in T
	STTorque                 = NewVectorField("STTorque", "T", AddSTTorque)
	J                        excitation // Polarized electrical current density
	MaxTorque                *GetScalar
	GammaLL                  float64 = 1.7595e11 // Gyromagnetic ratio of spins, in rad/Ts
	Precess                          = true
	DisableZhangLiTorque             = false
	DisableSlonczewskiTorque         = false
	// For first additional source of spin torque
	DisableSlonczewskiTorque1        = false
	Pfree1			 ScalarParam
	Pfixed1			 ScalarParam
	Lambdafree1		 ScalarParam
	Lambdafixed1		 ScalarParam
	EpsilonPrime1		 ScalarParam
	FixedLayer1		 VectorParam
	Jint1			 excitation // Polarized electrical current density
	// For second additional source of spin torque
	DisableSlonczewskiTorque2        = false
	Pfree2			 ScalarParam
	Pfixed2			 ScalarParam
	Lambdafree2		 ScalarParam
	Lambdafixed2		 ScalarParam
	EpsilonPrime2		 ScalarParam
	FixedLayer2		 VectorParam
	Jint2			 excitation // Polarized electrical current density
)

func init() {
	Export(STTorque, "Spin-transfer torque/γ0")

	Alpha.init("alpha", "", "Landau-Lifshitz damping constant", []derived{&temp_red})
	Xi.init("xi", "", "Non-adiabaticity of spin-transfer-torque", nil)
	J.init("J", "A/m2", "Electrical current density")
	Pol.init("Pol", "", "Electrical current polarization", nil)
	Pol.setUniform([]float64{1}) // default spin polarization
	Lambda.init("Lambda", "", "Slonczewski Λ parameter", nil)
	Lambda.Set(1) // sensible default value (?). TODO: should not be zero
	EpsilonPrime.init("EpsilonPrime", "", "Slonczewski secondairy STT term ε'", nil)
	FrozenSpins.init("frozenspins", "", "Defines spins that should be fixed", nil) // 1 - frozen, 0 - free. TODO: check if it only contains 0/1 values
	FixedLayer.init("FixedLayer", "", "Slonczewski fixed layer polarization")
	LLTorque.init("LLtorque", "T", "Landau-Lifshitz torque/γ0", SetLLTorque)
	Torque.init("torque", "T", "Total torque/γ0", SetTorque)
	DeclVar("GammaLL", &GammaLL, "Gyromagnetic ratio in rad/Ts")
	DeclVar("DisableZhangLiTorque", &DisableZhangLiTorque, "Disables Zhang-Li torque (default=false)")
	DeclVar("DisableSlonczewskiTorque", &DisableSlonczewskiTorque, "Disables Slonczewski torque (default=false)")
	DeclVar("DoPrecess", &Precess, "Enables LL precession (default=true)")
	MaxTorque = NewGetScalar("maxTorque", "T", "Maximum torque/γ0, over all cells", GetMaxTorque)
	// For first additional interface
	DeclVar("DisableSlonczewskiTorque1", &DisableSlonczewskiTorque1, "Disables Slonczewski torque from first additional interface (default=true)")
	Jint1.init("Jint1", "A/m2", "Electrical current density")
	Pfree1.init("Pfree1", "", "Electrical current polarization", nil)
	Pfree1.setUniform([]float64{1}) // default spin polarization
	Pfixed1.init("Pfix1", "", "Electrical current polarization", nil)
	Pfixed1.setUniform([]float64{1}) // default spin polarization
	Lambdafree1.init("Lambdafree1", "", "Slonczewski Λ parameter", nil)
	Lambdafree1.Set(1) // sensible default value (?). TODO: should not be zero
	Lambdafixed1.init("Lambdafix1", "", "Slonczewski Λ parameter", nil)
	Lambdafixed1.Set(1) // sensible default value (?). TODO: should not be zero
	EpsilonPrime1.init("EpsilonPrime1", "", "Slonczewski secondairy STT term ε'", nil)
	FixedLayer1.init("FixedLayer1", "", "Slonczewski fixed layer polarization")
	// For second additional interface
	DeclVar("DisableSlonczewskiTorque2", &DisableSlonczewskiTorque2, "Disables Slonczewski torque from second additional interface (default=true)")
	Jint2.init("Jint2", "A/m2", "Electrical current density")
	Pfree2.init("Pfree2", "", "Electrical current polarization", nil)
	Pfree2.setUniform([]float64{1}) // default spin polarization
	Pfixed2.init("Pfix2", "", "Electrical current polarization", nil)
	Pfixed2.setUniform([]float64{1}) // default spin polarization
	Lambdafree2.init("Lambdafree2", "", "Slonczewski Λ parameter", nil)
	Lambdafree2.Set(1) // sensible default value (?). TODO: should not be zero
	Lambdafixed2.init("Lambdafix2", "", "Slonczewski Λ parameter", nil)
	Lambdafixed2.Set(1) // sensible default value (?). TODO: should not be zero
	EpsilonPrime2.init("EpsilonPrime2", "", "Slonczewski secondairy STT term ε'", nil)
	FixedLayer2.init("FixedLayer2", "", "Slonczewski fixed layer polarization")
}

// Sets dst to the current total torque
func SetTorque(dst *data.Slice) {
	SetLLTorque(dst)
	AddSTTorque(dst)
	AddSTTorque1(dst)
	AddSTTorque2(dst)
	FreezeSpins(dst)
}

// Sets dst to the current Landau-Lifshitz torque
func SetLLTorque(dst *data.Slice) {
	B_eff.Set(dst) // calc and store B_eff
	if Precess {
		cuda.LLTorque(dst, M.Buffer(), dst, Alpha.gpuLUT1(), regions.Gpu()) // overwrite dst with torque
	} else {
		cuda.LLNoPrecess(dst, M.Buffer(), dst)
	}
}

// Adds the current spin transfer torque to dst
func AddSTTorque(dst *data.Slice) {
	if J.isZero() {
		return
	}
	util.AssertMsg(!Pol.isZero(), "spin polarization should not be 0")
	jspin, rec := J.Slice()
	if rec {
		defer cuda.Recycle(jspin)
	}
	if !DisableZhangLiTorque {
		cuda.AddZhangLiTorque(dst, M.Buffer(), jspin, Bsat.gpuLUT1(),
			Alpha.gpuLUT1(), Xi.gpuLUT1(), Pol.gpuLUT1(), regions.Gpu(), Mesh())
	}
	if !DisableSlonczewskiTorque && !FixedLayer.isZero() {
		cuda.AddSlonczewskiTorque(dst, M.Buffer(), jspin, FixedLayer.gpuLUT(), Msat.gpuLUT1(),
			Alpha.gpuLUT1(), Pol.gpuLUT1(), Lambda.gpuLUT1(), EpsilonPrime.gpuLUT1(), regions.Gpu(), Mesh())
	}
}

// Adds the current spin transfer torque from first additional source to dst
func AddSTTorque1(dst *data.Slice) {
	if Jint1.isZero() {
		return
	}
	util.AssertMsg(!Pfree1.isZero(), "interface 1: spin polarization (free) should not be 0")
	util.AssertMsg(!Pfixed1.isZero(), "interface 1: spin polarization (fixed) should not be 0")
	jspin, rec := Jint1.Slice()
	if rec {
		defer cuda.Recycle(jspin)
	}
	if !DisableSlonczewskiTorque1 && !FixedLayer1.isZero() {
		cuda.AddOommfSlonczewskiTorque(dst, M.Buffer(), jspin, FixedLayer1.gpuLUT(), Msat.gpuLUT1(),
			Alpha.gpuLUT1(), Pfixed1.gpuLUT1(), Pfree1.gpuLUT1(), Lambdafixed1.gpuLUT1(), Lambdafree1.gpuLUT1(), EpsilonPrime1.gpuLUT1(), regions.Gpu(), Mesh())
	}
}

// Adds the current spin transfer torque from second additional source to dst
func AddSTTorque2(dst *data.Slice) {
	if Jint2.isZero() {
		return
	}
	util.AssertMsg(!Pfree2.isZero(), "interface 1: spin polarization (free) should not be 0")
	util.AssertMsg(!Pfixed2.isZero(), "interface 1: spin polarization (fixed) should not be 0")
	jspin, rec := Jint2.Slice()
	if rec {
		defer cuda.Recycle(jspin)
	}
	if !DisableSlonczewskiTorque2 && !FixedLayer2.isZero() {
		cuda.AddOommfSlonczewskiTorque(dst, M.Buffer(), jspin, FixedLayer1.gpuLUT(), Msat.gpuLUT1(),
			Alpha.gpuLUT1(), Pfixed2.gpuLUT1(), Pfree2.gpuLUT1(), Lambdafixed2.gpuLUT1(), Lambdafree2.gpuLUT1(), EpsilonPrime2.gpuLUT1(), regions.Gpu(), Mesh())
	}
}

func FreezeSpins(dst *data.Slice) {
	if !FrozenSpins.isZero() {
		cuda.ZeroMask(dst, FrozenSpins.gpuLUT1(), regions.Gpu())
	}
}

// Gets
func GetMaxTorque() float64 {
	torque, recycle := Torque.Slice()
	if recycle {
		defer cuda.Recycle(torque)
	}
	return cuda.MaxVecNorm(torque)
}
