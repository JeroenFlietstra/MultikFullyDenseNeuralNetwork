package accuracy

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array

abstract class Accuracy(protected open val showConfusionMatrix: Boolean) {

    fun calculate(yPrediction: D1Array<Float>, yExpected: D1Array<Float>): Double {
        val comparisons = compare(yPrediction, yExpected)
        return mk.stat.mean(comparisons)
    }

    abstract fun compare(yPrediction: D1Array<Float>, yExpected: D1Array<Float>): D1Array<Float>
}