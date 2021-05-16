package network

import accuracy.Accuracy
import layer.Layer
import loss.Loss
import optimizer.Optimizer
import org.jetbrains.kotlinx.multik.api.d1array
import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.div
import toDimensionArray

class SequentialNetwork(private val loss: Loss, private val optimizer: Optimizer, private val accuracy: Accuracy) {

    private val layers = arrayListOf<Layer>()
    private var losses = mk.d2array(1, 1) { 0F }

    fun addLayer(layer: Layer) {
        layers.add(layer)
    }

    fun fit(x: D3Array<Float>, y: D1Array<Float>, numberOfEpochs: Int) {
        losses = mk.d2array(numberOfEpochs, x.shape[0]) { 0F }
        for (epoch in 1..numberOfEpochs) {
            for (i in 0 until x.shape[0]) {
                val yPred = predict(x[i].toDimensionArray())
                val yExp = y[i]

                losses[epoch - 1, i] = loss.compute(yPred[0], yExp)
                val e = loss.differentiate(yPred[0], yExp)

                for (layer in layers.reversed()) {
                    layer.propagateBackward(e, optimizer)
                }
            }

            println("Epoch: $epoch - Avg. Loss: ${mk.stat.mean(losses[epoch - 1])}")
        }
    }

    fun predict(x: D2Array<Float>): D1Array<Float> {
        var output = x
        for (layer in layers) {
            output = layer.propagateForward(output)
        }
        return output.flatten().asDNArray().asD1Array()
    }

    fun evaluate(x: D3Array<Float>, y: D1Array<Float>) {
        val yPrediction = mk.d1array(x.shape[0]) { 0F }

        for (i in 0 until x.shape[0]) {
            yPrediction[i] = predict(x[i].div(1F))[0]
        }

        println("\nAccuracy: ${accuracy.calculate(yPrediction, y) * 100}%")
    }
}