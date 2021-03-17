package main

import (
	"code/algo"
	"fmt"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	a := algo.Counter{Eps: 0.5, MaxValue: 1e3}
	fmt.Println(a.OneBit(999))
}
