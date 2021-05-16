package loss

import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min

sealed interface Loss {
    fun compute(yPred: Float, yExp: Float): Float
    fun differentiate(yPred: Float, yExp: Float): Float
}

object BinaryCrossEntropy : Loss {

    override fun compute(yPred: Float, yExp: Float): Float {
        val yPredClipped = clip(yPred)
        return if (yExp == 1F) {
            -log10(yPredClipped)
        } else {
            -log10(1 - yPredClipped)
        }
    }

    override fun differentiate(yPred: Float, yExp: Float): Float {
        val yPredClipped = clip(yPred)
        return if (yExp == 1F) {
            -1 / yPredClipped
        } else {
            1 / (1 - yPredClipped)
        }
    }

    /** Clip Float to prevent division by zero leading to a loss of Infinity */
    private fun clip(y: Float) = min(1 - 1E-7F, max(y, 1E-7F))
}