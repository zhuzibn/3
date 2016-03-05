package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// divide: dst[i] = a[i] / b[i]
// divide by zero automagically returns 0.0
func Divide(dst, a, b *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(a.Len() == N && a.NComp() == nComp && b.Len() == N && b.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_divide_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), N, cfg)
	}
}
