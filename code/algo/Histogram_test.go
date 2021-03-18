package algo

import (
	"math/rand"
	"testing"
	"time"
)

func TestNewClient(t *testing.T) {
	rand.Seed(time.Now().UnixNano())

	histo := Histogram{
		Eps:         1,
		NumBuckets:  1e3,
		PickBuckets: 1e2,
		MaxValue:    1e3,
	}
	c := NewClient(&histo)
	if len(c.Buckets) != histo.PickBuckets {
		t.Errorf("Client picked %d buckets, expected to pick %d", len(c.Buckets), histo.PickBuckets)
	}

	lookup := make(map[int]bool)
	for i := range c.Buckets {
		_, seen := lookup[c.Buckets[i]]
		if seen {
			t.Errorf("Client picked bucket %d twice", c.Buckets[i])
		}
		lookup[c.Buckets[i]] = true
	}
}

func TestHistogramEstimation(t *testing.T) {
	rand.Seed(time.Now().UnixNano())

	var eps float64 = 0.1
	var n int = 1e6
	histo := Histogram{
		Eps:         eps,
		NumBuckets:  1e2,
		PickBuckets: 1e1,
		MaxValue:    1e3,
	}
	buckets := make([]float64, histo.NumBuckets)
	bucketSize := histo.MaxValue / float64(histo.NumBuckets)
	clients := make([]Client, n)
	bits := make([][][2]int, n)
	for i := range clients {
		value := rand.Float64() * float64(histo.MaxValue)
		bucket := int(value / bucketSize)
		buckets[bucket] += 1.0 / float64(n)
		clients[i] = NewClient(&histo)
		bits[i] = clients[i].DBitFlip(value)
	}

	estimated := histo.Estimate(bits)
	t.Logf(`Epsilon %v:
	Actual histogram: %v
	Estimated histogram: %v`,
		eps,
		buckets,
		estimated,
	)
}
