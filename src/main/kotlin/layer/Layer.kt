package layer

import activation.Activation
import optimizer.Optimizer
import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array

abstract class Layer(shape: Pair<Int, Int>, val activation: Activation) {

    protected var weights: D2Array<Float>
    protected var biases: D2Array<Float>

    init {
        val numberOfInputs = shape.first
        val numberOfOutputs = shape.second
        weights = mk.d2array(numberOfOutputs, numberOfInputs) { (-0.1 + Math.random() * 0.2).toFloat() }
        biases = mk.d2array(numberOfOutputs, 1) { (-0.1 + Math.random() * 0.2).toFloat() }
    }

    lateinit var input: D2Array<Float>
    lateinit var output: D2Array<Float>

    abstract fun propagateForward(x: D2Array<Float>): D2Array<Float>

    abstract fun propagateBackward(e: Float, optimizer: Optimizer)
}