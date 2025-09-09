package com.example.onnxinfer
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import ai.onnxruntime.*
import android.graphics.Canvas
import android.graphics.Color
import java.nio.FloatBuffer
import java.nio.IntBuffer


class MainActivity : AppCompatActivity() {

    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var detectorSession: OrtSession
    private lateinit var decoderSession: OrtSession

    private val textThreshold = 0.7f
    private val linkThreshold = 0.4f
    private val lowText = 0.4f

    // EasyOCR detector constants
    private val canvasSize = 800
    private val magRatio =0.76f
    private val imgH = 64 // Recognition model height

    companion object {
        const val TAG = "ONNXSpeed"
        const val MODEL_NAME_DETECTOR = "model_combined_detector.onnx"
        const val MODEL_NAME_RECOG = "model_combined_recog.onnx"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        loadModel()
        testInference()
    }

    private fun loadModel() {
        val startTime = System.currentTimeMillis()

        try {
            ortEnvironment = OrtEnvironment.getEnvironment()
            val modelBytes = assets.open(MODEL_NAME_DETECTOR).readBytes()
            detectorSession = ortEnvironment.createSession(modelBytes)
            val decModelBytes = assets.open(MODEL_NAME_RECOG).readBytes()
            decoderSession = ortEnvironment.createSession(decModelBytes)
            val loadTime = System.currentTimeMillis() - startTime
            Log.d(TAG, "Model loaded in ${loadTime}ms")
            logModelIO()

        } catch (e: Exception) {
            Log.e(TAG, "Model load failed: ${e.message}")
        }
    }

    private fun logModelIO() {
        Log.d(TAG, "=== DETECTOR MODEL I/O INFO ===")
        detectorSession.inputInfo.forEach { (name, info) ->
            val tensorInfo = when (info) {
                is TensorInfo -> info
                is NodeInfo -> info.info as? TensorInfo
                else -> null
            }
            Log.d(TAG, "INPUT - Name: $name, Shape: ${tensorInfo?.shape?.contentToString()}")
        }

        detectorSession.outputInfo.forEach { (name, info) ->
            val tensorInfo = when (info) {
                is TensorInfo -> info
                is NodeInfo -> info.info as? TensorInfo
                else -> null
            }
            Log.d(TAG, "OUTPUT - Name: $name, Shape: ${tensorInfo?.shape?.contentToString()}")
        }

        Log.d(TAG, "=== RECOGNITION MODEL I/O INFO ===")
        decoderSession.inputInfo.forEach { (name, info) ->
            val tensorInfo = when (info) {
                is TensorInfo -> info
                is NodeInfo -> info.info as? TensorInfo
                else -> null
            }
            Log.d(TAG, "INPUT - Name: $name, Shape: ${tensorInfo?.shape?.contentToString()}")
        }

        decoderSession.outputInfo.forEach { (name, info) ->
            val tensorInfo = when (info) {
                is TensorInfo -> info
                is NodeInfo -> info.info as? TensorInfo
                else -> null
            }
            Log.d(TAG, "OUTPUT - Name: $name, Shape: ${tensorInfo?.shape?.contentToString()}")
        }
    }

    private fun testInference() {
        try {
            runCompleteOCRPipeline()
        } catch (e: Exception) {
            Log.e(TAG, "OCR pipeline test failed: ${e.message}")
        }
    }

    private fun runCompleteOCRPipeline() {
        try {
            // Load and process real image
            val bitmap = assets.open("easydemo.png").use {
                BitmapFactory.decodeStream(it)
            }

            Log.d(TAG, "Starting EasyOCR pipeline...")
            val startTime = System.currentTimeMillis()

            // Step 1: Preprocess for detector (EasyOCR style)
            val (detectorInput, ratios) = detectorPreprocess(bitmap)

            // Step 2: Run detector
            val detectorOutputs = runDetector(detectorInput)
            detectorInput.close()

            // Step 3: Post-process detector results to get text boxes
            val textBoxes = detectorPostprocess(detectorOutputs, ratios)
            detectorOutputs.forEach { it.value.close() }

            Log.d(TAG, "Found ${textBoxes.size} text boxes")

            // Step 4: Convert to grayscale for recognition
            val grayBitmap = convertToGrayscale(bitmap)

            // Step 5: Process each text box with recognition model
            val recognizedTexts = mutableListOf<RecognitionResult>()

            textBoxes.forEachIndexed { index, box ->
                try {
                    // Get image patches from the detected boxes
                    Log.d(TAG, "Box $index: ${box}")

                    val imagePatch = getImagePatch(grayBitmap, box)
                    val recognitionInput = recognizerPreprocess(imagePatch)
                    val recognitionOutput = runRecognition(recognitionInput)
                    val text = recognizerPostprocess(recognitionOutput)
                    recognitionInput.close()
                    recognitionOutput.forEach { it.value.close() }

                    if (text.text.isNotBlank() && text.confidence > 0.1f) {
                        recognizedTexts.add(RecognitionResult(box, text.text, text.confidence))
                        Log.d(TAG, "Box $index: '${text.text}' (conf: ${text.confidence})")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to process box $index: ${e.message}")
                }
            }

            val totalTime = System.currentTimeMillis() - startTime
            Log.d(TAG, "=== EasyOCR RESULTS ===")
            Log.d(TAG, "Total time: ${totalTime}ms")
            Log.d(TAG, "Detected boxes: ${textBoxes.size}")
            Log.d(TAG, "Recognized texts: ${recognizedTexts.size}")
            recognizedTexts.forEachIndexed { index, result ->
                Log.d(TAG, "Result $index: '${result.text}' (confidence: ${result.confidence})")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Complete EasyOCR pipeline failed: ${e.message}")
            e.printStackTrace()
        }
    }

    private fun detectorPreprocess(bitmap: Bitmap): Pair<OnnxTensor, Pair<Float, Float>> {
        // Convert bitmap to RGB array
        val rgbArray = bitmapToRgbArray(bitmap)

        // Resize with aspect ratio preservation (EasyOCR style)
        val (resizedImg, targetRatio) = resizeAspectRatio(rgbArray, canvasSize, magRatio)

        // Normalize using EasyOCR normalization
        val normalizedImg = normalizeMeanVariance(resizedImg)

        // Convert to tensor [1, 3, H, W]
        val height = normalizedImg.size
        val width = normalizedImg[0].size
        val channels = normalizedImg[0][0].size

        val floatArray = FloatArray(channels * height * width)

        // Transpose from HWC to CHW
        for (c in 0 until channels) {
            for (h in 0 until height) {
                for (w in 0 until width) {
                    floatArray[c * height * width + h * width + w] = normalizedImg[h][w][c]
                }
            }
        }

        val shape = longArrayOf(1, channels.toLong(), height.toLong(), width.toLong())
        val tensor = OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(floatArray), shape)

        val ratioH = 1.0f / targetRatio
        val ratioW = 1.0f / targetRatio

        return Pair(tensor, Pair(ratioH, ratioW))
    }

    private fun bitmapToRgbArray(bitmap: Bitmap): Array<Array<IntArray>> {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val rgbArray = Array(height) { Array(width) { IntArray(3) } }

        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = pixels[y * width + x]
                rgbArray[y][x][0] = (pixel shr 16) and 0xFF // R
                rgbArray[y][x][1] = (pixel shr 8) and 0xFF  // G
                rgbArray[y][x][2] = pixel and 0xFF          // B
            }
        }

        return rgbArray
    }

    private fun resizeAspectRatio(img: Array<Array<IntArray>>, canvasSize: Int, magRatio: Float): Pair<Array<Array<FloatArray>>, Float> {
        val height = img.size
        val width = img[0].size

        // Calculate target size maintaining aspect ratio
        val targetSize = (canvasSize * magRatio).toInt()
        val ratio = minOf(targetSize.toFloat() / width, targetSize.toFloat() / height)

        val targetWidth = 800
        val targetHeight = (height * ratio).toInt()
        Log.d(TAG, "current WIDTH: ${width}")

        Log.d(TAG, "current height: ${height}")


        Log.d(TAG, "target WIDTH: ${targetHeight}")

        Log.d(TAG, "target height: ${targetHeight}")


        // Simple resize (you might want to use better interpolation)
        val resized = Array(targetHeight) { Array(targetWidth) { FloatArray(3) } }

        for (y in 0 until targetHeight) {
            for (x in 0 until targetWidth) {
                val srcY = (y / ratio).toInt().coerceIn(0, height - 1)
                val srcX = (x / ratio).toInt().coerceIn(0, width - 1)

                resized[y][x][0] = img[srcY][srcX][0].toFloat()
                resized[y][x][1] = img[srcY][srcX][1].toFloat()
                resized[y][x][2] = img[srcY][srcX][2].toFloat()
            }
        }

        return Pair(resized, ratio)
    }

    private fun normalizeMeanVariance(img: Array<Array<FloatArray>>): Array<Array<FloatArray>> {
        val height = img.size
        val width = img[0].size
        val normalized = Array(height) { Array(width) { FloatArray(3) } }

        // EasyOCR normalization: (x - 127.5) / 127.5
        for (y in 0 until height) {
            for (x in 0 until width) {
                normalized[y][x][0] = (img[y][x][0] - 127.5f) / 127.5f
                normalized[y][x][1] = (img[y][x][1] - 127.5f) / 127.5f
                normalized[y][x][2] = (img[y][x][2] - 127.5f) / 127.5f
            }
        }

        return normalized
    }

    private fun runDetector(inputTensor: OnnxTensor): OrtSession.Result {
        val startTime = System.currentTimeMillis()
        val inputName = detectorSession.inputNames.first()
        val inputs = mapOf(inputName to inputTensor)
        val outputs = detectorSession.run(inputs)
        val inferenceTime = System.currentTimeMillis() - startTime

        Log.d(TAG, "Detector inference: ${inferenceTime}ms")

        // Log output shapes
        outputs.forEachIndexed { idx, output ->
            val tensor = output.value as OnnxTensor
            val shape = tensor.info.shape
            Log.d(TAG, "Detector output[$idx] shape: ${shape?.contentToString()}")
        }

        return outputs
    }

    private fun detectorPostprocess(outputs: OrtSession.Result, ratios: Pair<Float, Float>): List<TextBox> {
        val textBoxes = mutableListOf<TextBox>()

        try {
            // EasyOCR detector outputs score and link maps
            val scoreOutput = outputs[0] as OnnxTensor
            val scoreData = scoreOutput.floatBuffer.array()
            val scoreShape = scoreOutput.info.shape

            Log.d(TAG, "Processing detector output shape: ${scoreShape?.contentToString()}")

            // Assuming output is [1, H, W, 2] where channel 0 = text score, channel 1 = link score
            val height = scoreShape?.get(1)?.toInt() ?: return emptyList()
            val width = scoreShape?.get(2)?.toInt() ?: return emptyList()
            val channels = scoreShape?.get(3)?.toInt() ?: return emptyList()

            if (channels >= 2) {
                // Extract text and link scores
                val textScores = FloatArray(height * width)
                val linkScores = FloatArray(height * width)

                for (y in 0 until height) {
                    for (x in 0 until width) {
                        val baseIdx = y * width * channels + x * channels
                        textScores[y * width + x] = scoreData[baseIdx]
                        linkScores[y * width + x] = scoreData[baseIdx + 1]
                    }
                }

                // Simple connected component analysis to find text regions
                val textBoxesRaw = extractTextBoxes(textScores, linkScores, width, height, textThreshold, linkThreshold)

                // Adjust coordinates back to original image space
                // The scaling ratio was used to resize original -> 608x800
                // So we need to scale back: detector_coords / ratio = original_coords
                textBoxes.addAll(textBoxesRaw.map { box ->
                    TextBox(
                        (box.x1 / ratios.first).toInt(),
                        (box.y1 / ratios.second).toInt(),
                        (box.x2 / ratios.first).toInt(),
                        (box.y2 / ratios.second).toInt()
                    )
                })
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error in detector postprocess: ${e.message}")
            e.printStackTrace()
        }

        return textBoxes
    }

    private fun extractTextBoxes(textScores: FloatArray, linkScores: FloatArray, width: Int, height: Int, textThresh: Float, linkThresh: Float): List<TextBox> {
        val boxes = mutableListOf<TextBox>()
        val visited = BooleanArray(textScores.size)

        Log.d(TAG, "Extracting text boxes with thresholds: text=$textThresh, link=$linkThresh")

        // Analyze score distribution first
        val textAboveThresh = textScores.count { it > textThresh }
        val linkAboveThresh = linkScores.count { it > linkThresh }
        val maxText = textScores.maxOrNull() ?: 0f
        val maxLink = linkScores.maxOrNull() ?: 0f
        val avgText = textScores.average().toFloat()
        val avgLink = linkScores.average().toFloat()

        Log.d(TAG, "Score analysis - Text: max=$maxText, avg=$avgText, above_thresh=$textAboveThresh")
        Log.d(TAG, "Score analysis - Link: max=$maxLink, avg=$avgLink, above_thresh=$linkAboveThresh")

        // If very few pixels above threshold, try lower thresholds
        var actualTextThresh = textThresh
        var actualLinkThresh = linkThresh

        if (textAboveThresh < 10) {
            actualTextThresh = minOf(textThresh, maxText * 0.5f)
            Log.w(TAG, "Very few text pixels, lowering threshold to $actualTextThresh")
        }

        if (linkAboveThresh < 10) {
            actualLinkThresh = minOf(linkThresh, maxLink * 0.5f)
            Log.w(TAG, "Very few link pixels, lowering threshold to $actualLinkThresh")
        }

        // Simple connected component labeling
        var componentsFound = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val idx = y * width + x
                if (textScores[idx] > actualTextThresh && !visited[idx]) {
                    val component = mutableListOf<Pair<Int, Int>>()
                    val queue = mutableListOf<Pair<Int, Int>>()
                    queue.add(Pair(x, y))
                    visited[idx] = true

                    while (queue.isNotEmpty()) {
                        val (cx, cy) = queue.removeAt(0)
                        component.add(Pair(cx, cy))

                        // Check 8-connected neighbors
                        for (dy in -1..1) {
                            for (dx in -1..1) {
                                if (dx == 0 && dy == 0) continue
                                val nx = cx + dx
                                val ny = cy + dy
                                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                                    val nidx = ny * width + nx
                                    if (!visited[nidx] && (textScores[nidx] > actualTextThresh || linkScores[nidx] > actualLinkThresh)) {
                                        visited[nidx] = true
                                        queue.add(Pair(nx, ny))
                                    }
                                }
                            }
                        }
                    }

                    componentsFound++
                    Log.d(TAG, "Component $componentsFound: size=${component.size}")

                    if (component.size >= 5) { // Even smaller minimum component size
                        val minX = component.minOf { it.first }
                        val maxX = component.maxOf { it.first }
                        val minY = component.minOf { it.second }
                        val maxY = component.maxOf { it.second }

                        val box = TextBox(minX, minY, maxX + 1, maxY + 1) // Add 1 for inclusive bounds
                        boxes.add(box)
                        Log.d(TAG, "Added box: ($minX,$minY) to ($maxX,$maxY) size=${maxX-minX}x${maxY-minY}")
                    } else {
                        Log.d(TAG, "Rejected small component: size=${component.size}")
                    }
                }
            }
        }

        Log.d(TAG, "Found $componentsFound components, kept ${boxes.size} boxes")
        return boxes
    }

    // Data classes
    data class TextBox(val x1: Int, val y1: Int, val x2: Int, val y2: Int)
    data class TextResult(val text: String, val confidence: Float)
    data class RecognitionResult(val box: TextBox, val text: String, val confidence: Float)

    private fun convertToGrayscale(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val grayBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            val gray = (0.299 * r + 0.587 * g + 0.114 * b).toInt()
            pixels[i] = (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
        }

        grayBitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return grayBitmap
    }

    private fun getImagePatch(grayBitmap: Bitmap, box: TextBox): Bitmap {
        val left = maxOf(0, box.x1)
        val top = maxOf(0, box.y1)
        val right = minOf(grayBitmap.width, box.x2)
        val bottom = minOf(grayBitmap.height, box.y2)
        Log.d(TAG, "bbox shape ahh shape: ${box}}")

        val width = right - left
        val height = top - bottom
        Log.d(TAG, "HEIGHT: ${height}")

        Log.d(TAG, "WIDTH: ${width}")

        if (width <= 0 || height <= 0) {
            throw IllegalArgumentException("Invalid box dimensions")
        }

        return Bitmap.createBitmap(grayBitmap, left, top, width, height)
    }

    private fun recognizerPreprocess(imagePatch: Bitmap): OnnxTensor {
        // EasyOCR recognition preprocessing: resize to height 64, keep aspect ratio
        val targetHeight = imgH
        val aspectRatio = imagePatch.width.toFloat() / imagePatch.height.toFloat()
        val targetWidth = (targetHeight * aspectRatio).toInt()

        // Pad to fixed width if needed (your model expects 1000 width)
        val paddedWidth = 1000
        val resized = Bitmap.createScaledBitmap(imagePatch, targetWidth, targetHeight, true)

        // Create padded image
        val padded = Bitmap.createBitmap(paddedWidth, targetHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(padded)
        canvas.drawColor(Color.WHITE) // White background
        canvas.drawBitmap(resized, 0f, 0f, null)

        // Convert to tensor [1, 1, 64, 1000]
        val pixels = IntArray(paddedWidth * targetHeight)
        padded.getPixels(pixels, 0, paddedWidth, 0, 0, paddedWidth, targetHeight)

        val floatArray = FloatArray(paddedWidth * targetHeight)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val gray = (pixel shr 16) and 0xFF // Already grayscale, just take R channel
            floatArray[i] = (gray - 127.5f) / 127.5f // Normalize to [-1, 1]
        }

        val shape = longArrayOf(1, 1, targetHeight.toLong(), paddedWidth.toLong())
        return OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(floatArray), shape)
    }

    private fun runRecognition(inputTensor: OnnxTensor): OrtSession.Result {
        val startTime = System.currentTimeMillis()
        val inputName = decoderSession.inputNames.first()
        val inputs = mapOf(inputName to inputTensor)
        val outputs = decoderSession.run(inputs)
        val inferenceTime = System.currentTimeMillis() - startTime

        Log.d(TAG, "Recognition inference: ${inferenceTime}ms")

        outputs.forEachIndexed { idx, output ->
            val tensor = output.value as OnnxTensor
            val shape = tensor.info.shape
            Log.d(TAG, "Recognition output[$idx] shape: ${shape?.contentToString()}")
        }

        return outputs
    }

    private fun recognizerPostprocess(outputs: OrtSession.Result): TextResult {
        try {
            val output = outputs[0] as OnnxTensor
            val outputData = output.floatBuffer.array()
            val outputShape = output.info.shape

            // Assuming output is [batch, sequence_length, vocab_size]
            val batchSize = outputShape?.get(0)?.toInt() ?: 1
            val seqLength = outputShape?.get(1)?.toInt() ?: 0
            val vocabSize = outputShape?.get(2)?.toInt() ?: 0

            val text = StringBuilder()
            val confidenceScores = mutableListOf<Float>()

            // Greedy decoding (like EasyOCR)
            for (i in 0 until seqLength) {
                var maxProb = -Float.MAX_VALUE
                var maxIndex = 0

                for (j in 0 until vocabSize) {
                    val prob = outputData[i * vocabSize + j]
                    if (prob > maxProb) {
                        maxProb = prob
                        maxIndex = j
                    }
                }

                if (maxIndex > 0) { // Skip blank token (index 0)
                    val char = indexToChar(maxIndex)
                    if (char.isNotEmpty()) {
                        text.append(char)
                        confidenceScores.add(maxProb)
                    }
                }
            }

            val avgConfidence = if (confidenceScores.isNotEmpty()) {
                confidenceScores.average().toFloat()
            } else {
                0.0f
            }

            return TextResult(text.toString(), avgConfidence)

        } catch (e: Exception) {
            Log.e(TAG, "Error in recognition postprocess: ${e.message}")
            return TextResult("", 0.0f)
        }
    }

    private fun indexToChar(index: Int): String {
        // Basic EasyOCR character set - you'll need to match your model's actual vocabulary
        val chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
        return if (index > 0 && index <= chars.length) {
            chars[index - 1].toString()
        } else {
            ""
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::detectorSession.isInitialized) detectorSession.close()
        if (::decoderSession.isInitialized) decoderSession.close()
        if (::ortEnvironment.isInitialized) ortEnvironment.close()
    }

}