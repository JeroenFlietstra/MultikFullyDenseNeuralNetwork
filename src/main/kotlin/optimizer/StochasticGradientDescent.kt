package optimizer

import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.times

class StochasticGradientDescent(override val learningRate: Float = 0.01F) : Optimizer(learningRate) {

    override fun optimize(gradient: D2Array<Float>): D2Array<Float> {
        return gradient * -learningRate
    }
}