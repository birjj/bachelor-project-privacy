package algo

import (
	"math"
	"math/rand"
)

type Counter struct {
	Eps      float64
	MaxValue float64
}

// Implementation of 1BitMean
// Returns a bit indicating the value of the counter
func (c Counter) OneBit(value float64) byte {
	/*eEps := math.Exp(a.Eps)
	offset := 1 / (eEps + 1)
	frac := value / a.CounterMax
	scale := (eEps - 1) / (eEps + 1)
	probability := offset + frac*scale*/

	// rewriting of formula to avoid tiny float values
	eEps := math.Exp(c.Eps)
	top := c.MaxValue + value*eEps - value
	bot := c.MaxValue*eEps + c.MaxValue
	probability := top / bot
	randVal := rand.Float64()

	if randVal <= probability {
		return 1
	}
	return 0
}

// Implementation of mean estimation
// Returns the estimated mean from a set of bits gathered from 1BitMean
func (c Counter) Mean(bits []byte) float64 {
	eEps := math.Exp(c.Eps)
	n := len(bits)
	sum := 0.0
	for _, val := range bits {
		sum += (float64(val)*(eEps+1) - 1) / (eEps - 1)
	}
	return (c.MaxValue / float64(n)) * sum
}
