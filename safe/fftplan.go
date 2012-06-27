package safe

// INTERNAL
// Base implementation for all FFT plans.

import (
	"github.com/barnex/cuda4/cu"
	"github.com/barnex/cuda4/cufft"
)

// Base implementation for all FFT plans.
type fftplan struct {
	handle cufft.Handle
	stream cu.Stream
}

// For the sake of embedding.
type size1D int

// Returns the logical size of the FFT:
// the number of elements (real or complex)
// it transforms.
func (s size1D) Size() int { return int(s) }

// Releases all resources associated with the FFT plan.
func (p fftplan) Destroy() { p.handle.Destroy() }

// Associates a CUDA stream with the FFT plan.
// If a stream is set, plan.Stream().Synchronize() can
// to be called to wait for the execution to finish.
func (p fftplan) SetStream(stream cu.Stream) {
	p.handle.SetStream(stream)
	p.stream = stream
}

// Returns the CUDA stream associated with the FFT plan.
func (p fftplan) Stream() cu.Stream {
	return p.stream
}
