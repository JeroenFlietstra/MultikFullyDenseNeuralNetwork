package optimizer

import org.jetbrains.kotlinx.multik.ndarray.data.D2Array

abstract class Optimizer(open val learningRate: Float) {

    abstract fun optimize(gradient: D2Array<Float>): D2Array<Float>
}