package accuracy

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.mapIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

class Categorical(override val showConfusionMatrix: Boolean = false, private val isBinary: Boolean = false) :
    Accuracy(showConfusionMatrix) {

    override fun compare(yPrediction: D1Array<Float>, yExpected: D1Array<Float>): D1Array<Float> {
        return if (isBinary) {
            if (showConfusionMatrix) {
                val truePositives = yPrediction.toList().withIndex().count { it.value == 1F && yExpected[it.index] == 1F }
                val falsePositives = yPrediction.toList().withIndex().count { it.value == 1F && yExpected[it.index] == 0F }
                val trueNegatives = yPrediction.toList().withIndex().count { it.value == 0F && yExpected[it.index] == 0F }
                val falseNegatives = yPrediction.toList().withIndex().count { it.value == 0F && yExpected[it.index] == 1F }

                println("\nConfusion matrix:")
                println("${mk.ndarray(listOf(truePositives, falsePositives, falseNegatives, trueNegatives), 2, 2)}")
            }

            yPrediction.mapIndexed { index: Int, pred: Float ->
                ((pred >= .5F).compareTo(false).toFloat() == yExpected[index]).compareTo(false).toFloat()
            }
        } else {
            throw NotImplementedError()
        }
    }
}