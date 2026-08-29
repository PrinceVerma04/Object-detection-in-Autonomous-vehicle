package com.civicservices.monitor

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

data class Detection(val classId: Int, val label: String, val confidence: Float, val box: RectF)

/**
 * Wraps a YOLO11-family TFLite export (output shape [1, 4+numClasses, numAnchors], no
 * objectness channel) with manual decode + NMS, since a raw Interpreter export has no
 * bundled metadata for the TFLite Task library to use automatically.
 */
class Detector(context: Context, modelPath: String = "model.tflite") {

    val labels = listOf(
        "garbage_pile",
        "pothole",
        "manhole_open_broken",
        "streetlight_pole",
        "encroachment",
        "billboard_hoarding",
        "cattle",
    )

    private val interpreter: Interpreter
    private val inputSize: Int
    private val nchw: Boolean // Ultralytics' TFLite export is [1,3,H,W] (channels-first), not the [1,H,W,3] TFLite default
    private val numClasses: Int
    private val numAnchors: Int

    init {
        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, Interpreter.Options().apply { setNumThreads(4) })

        val inputShape = interpreter.getInputTensor(0).shape()
        nchw = inputShape[1] == 3
        inputSize = if (nchw) inputShape[2] else inputShape[1]

        val outputShape = interpreter.getOutputTensor(0).shape() // [1, 4+numClasses, numAnchors]
        numClasses = outputShape[1] - 4
        numAnchors = outputShape[2]
    }

    fun detect(bitmap: Bitmap, confThreshold: Float = 0.4f, iouThreshold: Float = 0.45f): List<Detection> {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val input = bitmapToByteBuffer(resized)
        val output = Array(1) { Array(4 + numClasses) { FloatArray(numAnchors) } }
        try {
            interpreter.run(input, output)
        } catch (e: IllegalStateException) {
            // Interpreter was closed mid-flight by a concurrent mode switch (e.g. starting a
            // video recording) — drop this one frame instead of crashing the whole app.
            return emptyList()
        }

        // Ultralytics TFLite exports sometimes emit box coords normalized to [0,1] and
        // sometimes in input-pixel space depending on export flags — detect which at runtime.
        var maxCoord = 0f
        for (a in 0 until numAnchors) {
            maxCoord = max(maxCoord, max(output[0][0][a], output[0][1][a]))
        }
        val coordScale = if (maxCoord <= 1.5f) inputSize.toFloat() else 1f

        val scaleX = bitmap.width.toFloat() / inputSize
        val scaleY = bitmap.height.toFloat() / inputSize

        val candidates = mutableListOf<Detection>()
        for (a in 0 until numAnchors) {
            var bestClass = -1
            var bestScore = 0f
            for (c in 0 until numClasses) {
                val score = output[0][4 + c][a]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c
                }
            }
            if (bestScore < confThreshold) continue

            val cx = output[0][0][a] * coordScale * scaleX
            val cy = output[0][1][a] * coordScale * scaleY
            val w = output[0][2][a] * coordScale * scaleX
            val h = output[0][3][a] * coordScale * scaleY

            val box = RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            candidates.add(Detection(bestClass, labels.getOrElse(bestClass) { "class_$bestClass" }, bestScore, box))
        }

        return nonMaxSuppression(candidates, iouThreshold)
    }

    private fun nonMaxSuppression(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val kept = mutableListOf<Detection>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            kept.add(best)
            sorted.removeAll { it.classId == best.classId && iou(best.box, it.box) > iouThreshold }
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        val interArea = max(0f, interRight - interLeft) * max(0f, interBottom - interTop)
        val union = a.width() * a.height() + b.width() * b.height() - interArea
        return if (union <= 0f) 0f else interArea / union
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        buffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        if (nchw) {
            // channel-planar: all R, then all G, then all B
            for (channelShift in intArrayOf(16, 8, 0)) {
                for (pixel in pixels) {
                    buffer.putFloat(((pixel shr channelShift) and 0xFF) / 255f)
                }
            }
        } else {
            // interleaved per pixel: R,G,B, R,G,B, ...
            for (pixel in pixels) {
                buffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
                buffer.putFloat(((pixel shr 8) and 0xFF) / 255f)
                buffer.putFloat((pixel and 0xFF) / 255f)
            }
        }
        buffer.rewind()
        return buffer
    }

    fun close() = interpreter.close()
}
