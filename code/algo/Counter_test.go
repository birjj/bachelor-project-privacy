package algo

import (
	"math/rand"
	"testing"
	"time"
)

func TestMeanEstimation(t *testing.T) {
	rand.Seed(time.Now().UnixNano())

	var n int = 1e6
	var eps float64 = 0.1
	var m float64 = 1e3
	a := Counter{Eps: eps, MaxValue: m}
	values := make([]float64, n)
	bits := make([]byte, n)
	for i := 0; i < n; i++ {
		values[i] = rand.Float64() * m
		bits[i] = a.OneBit(values[i])
	}

	actualSum := 0.0
	for _, val := range values {
		actualSum += val
	}
	t.Logf(`Epsilon %v:
	Actual mean: %v
	Estimated mean: %v`,
		eps,
		actualSum/float64(n),
		a.Mean(bits[:]),
	)
}
