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
    // Create data.
    generateData(1000)

    // Split data (70% train data, 30% test data).
    val total = x.shape[0]
    val limit = (total * 0.7).toInt()
    val trainX = x[Slice(0, limit, 1)].toDimensionArray()
    val testX = x[Slice(limit, total, 1)].toDimensionArray()
    val trainY = y[Slice(0, limit, 1)].toDimensionArray()
    val testY = y[Slice(limit, total, 1)].toDimensionArray()

    // Initialize model.
    val model = SequentialNetwork(
        loss = BinaryCrossEntropy,
        optimizer = StochasticGradientDescent(0.001F),
        accuracy = Categorical(isBinary = true, showConfusionMatrix = true)
    )

    // Create layers 3 input values → 4 hidden neurons → 1 output neuron.
    model.addLayer(Dense(shape = Pair(3, 4), activation = ReLU))
    model.addLayer(Dense(shape = Pair(4, 1), activation = Sigmoid))

    // Train model.
    model.fit(trainX, trainY, 10)

    // Test model.
    model.evaluate(testX, testY)
}

/**
 * Generate data consisting of 3 Float values between -10 and 10. These are the features (x), the target (y) consists
 * of byte representations of booleans whether the sum of these three Float values is positive or negative.
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