package com.example.facebluropencv

import android.content.res.AssetManager
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.FaceDetectorYN
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max

class Process {
    private lateinit var yunet: FaceDetectorYN

    companion object {
        const val MODEL = "face_detection_yunet_2023mar.onnx"
        const val SIZE = 640
        const val CONFIDENCE_THRESHOLD = 0.4f
        const val NMS_THRESHOLD = 0.4f
    }

    fun loadModel(assets: AssetManager, fileDir: String) {
        val outputFile = File("$fileDir/$MODEL")
        assets.open(MODEL).use { inputStream ->
            FileOutputStream(outputFile).use { outputStream ->
                val buffer = ByteArray(1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
            }
        }

        yunet = FaceDetectorYN.create(
            "$fileDir/$MODEL",
            "",
            Size(SIZE.toDouble(), SIZE.toDouble()),
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD
        )

    }

    fun detect(input: Mat): Mat {
        val inputMat = preProcess(input)
        val outputMat = Mat()

        yunet.detect(inputMat, outputMat)
        return outputMat
    }

    private fun preProcess(mat: Mat): Mat {
        val input = Mat()
        Imgproc.cvtColor(mat, input, Imgproc.COLOR_RGBA2RGB)
        Imgproc.resize(input, input, yunet.inputSize)
        input.convertTo(input, CvType.CV_32FC3)

        return input
    }

    fun postProcess(input: Mat, result: Mat?): Mat {
        if (result == null || result.total() == 0.toLong()) return input

        (0 until result.rows()).forEach {
            val dx = input.width() / SIZE.toFloat()
            val dy = input.height() / SIZE.toFloat()

            val left = max(0, (result.get(it, 0)[0] * dx).toInt())
            val top = max(0, (result.get(it, 1)[0] * dy).toInt())
            var width = (result.get(it, 2)[0] * dx).toInt()
            var height = (result.get(it, 3)[0] * dy).toInt()

            if (left + width > input.width()) width = input.width() - left
            if(top + height > input.height()) height = input.height() - top

            val rect = Rect(left, top, width, height)
            val rectColor = Scalar(0.0, 255.0, 255.0)

            val conf = result.get(it, 14)[0]
            val text = "%.2f".format(conf * 100).plus("%")
            val point = Point(left.toDouble(), top.toDouble() - 5)
            val font = Imgproc.FONT_HERSHEY_SIMPLEX
            val textColor = Scalar(0.0, 0.0, 0.0)

            val face = Mat(input, rect)
            Imgproc.GaussianBlur(face, face, Size(99.0, 99.0), 0.0, 0.0)
            Imgproc.rectangle(input, rect, rectColor, 3)
            Imgproc.putText(input, text, point, font, 1.3, textColor, 3)
        }
        return input
    }
}