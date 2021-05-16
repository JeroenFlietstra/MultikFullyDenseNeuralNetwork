import accuracy.Categorical
import activation.ReLU
import activation.Sigmoid
import layer.Dense
import loss.BinaryCrossEntropy
import network.SequentialNetwork
import optimizer.StochasticGradientDescent
import org.jetbrains.kotlinx.multik.api.d1array
import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.d3array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.div
import org.jetbrains.kotlinx.multik.ndarray.operations.sum

private lateinit var x: D3Array<Float>
private lateinit var y: D1Array<Float>

fun main() {
    generateData(1000)

    val total = x.shape[0]
    val limit = (total * 0.7).toInt()
    val trainX = x[Slice(0, limit, 1)].toDimensionArray()
    val testX = x[Slice(limit, total, 1)].toDimensionArray()
    val trainY = y[Slice(0, limit, 1)].toDimensionArray()
    val testY = y[Slice(limit, total, 1)].toDimensionArray()

    val model = SequentialNetwork(
        loss = BinaryCrossEntropy,
        optimizer = StochasticGradientDescent(0.001F),
        accuracy = Categorical(isBinary = true, showConfusionMatrix = true)
    )
    model.addLayer(Dense(shape = Pair(3, 4), activation = ReLU))
    model.addLayer(Dense(shape = Pair(4, 1), activation = Sigmoid))
    model.fit(trainX, trainY, 10)
    model.evaluate(testX, testY)
}

/**
 *
 */
fun generateData(numOfEntries: Int) {
    x = mk.d3array(numOfEntries, 3, 1) { 0F }
    y = mk.d1array(numOfEntries) { 0F }

    for (i in (0 until numOfEntries)) {
        val a = mk.d2array(3, 1) { 0F }

        while (a.sum() == 0F) {
            for (b in (0 until 3)) {
                a[b] = mk.d1array(1) { (-10..10).random().toFloat() }
            }
        }

        x[i] = a
        y[i] = (a.sum() >= 0).compareTo(false).toFloat()
    }

    println("Total inputs with positive sum: ${y.sum().toInt()}")
    println("Total inputs with negative sum: ${x.shape[0] - y.sum().toInt()}\n")
}

/**
 * Easy cast from MultiArray<Float, Dimension> to NDArray<Float, Dimension>. Seems to be missing in Multik.
 */
fun <D : Dimension> MultiArray<Float, D>.toDimensionArray() = div(1F)