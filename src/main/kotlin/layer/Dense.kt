package layer

import activation.Activation
import optimizer.Optimizer
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.times

class Dense(shape: Pair<Int, Int>, activation: Activation) : Layer(shape, activation) {

    override fun propagateForward(x: D2Array<Float>): D2Array<Float> {
        input = x
        output = activation.compute(mk.linalg.dot(weights, x) + biases)
        return output
    }

    override fun propagateBackward(e: Float, optimizer: Optimizer) {
        val gradientsWeights = mk.linalg.dot(activation.differentiate(output) * e, input.transpose())
        val gradientsBiases = activation.differentiate(output) * e
        weights += optimizer.optimize(gradientsWeights)
        biases += optimizer.optimize(gradientsBiases)
    }
}