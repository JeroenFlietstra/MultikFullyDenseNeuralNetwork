package activation

import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.*

sealed interface Activation {
    fun compute(x: D2Array<Float>): D2Array<Float>
    fun differentiate(x: D2Array<Float>): D2Array<Float>
}

object ReLU : Activation {

    override fun compute(x: D2Array<Float>): D2Array<Float> {
        return x.map { a -> if (a > 0) a else 0F }
    }

    override fun differentiate(x: D2Array<Float>): D2Array<Float> {
        return x.map { a -> (a > 0).compareTo(false).toFloat() }
    }
}

object Sigmoid : Activation {

    override fun compute(x: D2Array<Float>): D2Array<Float> {
        return mk.d2array(1, 1) { 1F } / (mk.math.exp(x * -1f) + 1.0).asType()
    }

    override fun differentiate(x: D2Array<Float>): D2Array<Float> {
        val s = compute(x)
        return s * (mk.d2array(1, 1) { 1F } - s)
    }
}
