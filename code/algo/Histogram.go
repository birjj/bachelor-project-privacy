package algo

import (
	"math"
	"math/rand"
)

type Histogram struct {
	Eps         float64
	NumBuckets  int
	PickBuckets int
	MaxValue    float64
}

type Client struct {
	Histogram *Histogram
	Buckets   []int
}

func NewClient(histogram *Histogram) Client {
	c := Client{
		Histogram: histogram,
		Buckets:   make([]int, histogram.PickBuckets),
	}

	// pick d buckets without replacement
	seen := make(map[int]bool)
	for i := range c.Buckets {
		var n int
		alreadyUsed := true
		for alreadyUsed {
			n = rand.Intn(histogram.NumBuckets)
			_, alreadyUsed = seen[n]
		}
		seen[n] = true
		c.Buckets[i] = n
	}

	return c
}

// Implementation of dBitFlip
// Returns d bits, one for each of the client's picked buckets, indicating whether the value is in said bucket
// Each bit is returned as the pair [bucket, bit] where bucket is the bucket index
func (c Client) DBitFlip(value float64) [][2]int {
	histo := c.Histogram
	bucketSize := histo.MaxValue / float64(histo.NumBuckets)
	bucket := int(value / bucketSize)

	bits := make([][2]int, len(c.Buckets))
	eEpsHalf := math.Exp(c.Histogram.Eps / 2)
	for i := range bits {
		top := 1.0
		if c.Buckets[i] == bucket {
			top = eEpsHalf
		}
		bot := eEpsHalf + 1
		probability := top / bot
		randVal := rand.Float64()

		pair := [2]int{c.Buckets[i], 0}
		if randVal <= probability {
			pair[1] = 1
		}
		bits[i] = pair
	}

	return bits
}

// Implementation of histogram estimation
func (h Histogram) Estimate(responses [][][2]int) []float64 {
	histogram := make([]float64, h.NumBuckets)
	eEpsHalf := math.Exp(h.Eps / 2)
	scale := float64(h.NumBuckets) / float64(len(responses)*h.PickBuckets)
	for j := range responses {
		bits := responses[j]
		for i := range bits {
			bucket := bits[i][0]
			bit := bits[i][1]

			top := float64(bit)*(eEpsHalf+1) - 1
			bot := eEpsHalf - 1
			histogram[bucket] += scale * top / bot
		}
	}

	return histogram
}
