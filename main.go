package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"
)

type Perceptron struct {
	input        [][]float64
	targetOutput []float64
	weights      []float64
	bias         float64
	epochs       int
}

func vectorDotProduct(v1, v2 []float64) float64 {
	if len(v1) != len(v2) {
		log.Fatalln("Length of vector 1 and vector 2 must be the same.")
	}
	dot := 0.0
	for i := 0; i < len(v1); i++ {
		dot += v1[i] * v2[i]
	}
	return dot
}

func vectorAdd(v1, v2 []float64) []float64 {
	if len(v1) != len(v2) {
		log.Fatalln("Length of vector 1 and vector 2 must be the same.")
	}
	add := make([]float64, len(v1))
	for i := 0; i < len(v1); i++ {
		add[i] = v1[i] + v2[i]
	}
	return add
}

func scalarMatMul(s float64, v []float64) []float64 {
	result := make([]float64, len(v))
	for i := 0; i < len(v); i++ {
		result[i] = s * v[i]
	}
	return result
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (p *Perceptron) initialize() {
	rand.Seed(time.Now().UnixNano())
	p.bias = 0.0
	p.weights = make([]float64, len(p.input[0]))
	for i := 0; i < len(p.input[0]); i++ {
		p.weights[i] = rand.Float64()
	}
}

func (p *Perceptron) forwardPass(x []float64) (sum float64) {
	return sigmoid(vectorDotProduct(p.weights, x) + p.bias)
}

func (p *Perceptron) gradW(x []float64, y float64) []float64 {
	pred := p.forwardPass(x)
	return scalarMatMul(-(pred-y)*pred*(1-pred), x)
}

func (p *Perceptron) gradB(x []float64, y float64) float64 {
	pred := p.forwardPass(x)
	return -(pred - y) * pred * (1 - pred)
}

func (p *Perceptron) train() {
	for i := 0; i < p.epochs; i++ {
		dw := make([]float64, len(p.input[0]))
		db := 0.0
		for length, val := range p.input {
			dw = vectorAdd(dw, p.gradW(val, p.targetOutput[length]))
			db += p.gradB(val, p.targetOutput[length])
		}
		dw = scalarMatMul(2/float64(len(p.targetOutput)), dw)
		p.weights = vectorAdd(p.weights, dw)
		p.bias += db * 2 / float64(len(p.targetOutput))
	}
}

func main() {
	data := Perceptron{
		input:        [][]float64{{0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 0}},
		targetOutput: []float64{0, 1, 1, 0},
		epochs:       1000,
	}
	data.initialize()
	data.train()
	fmt.Println(data.forwardPass([]float64{0, 1, 0}))
	fmt.Println(data.forwardPass([]float64{1, 0, 1}))
}
