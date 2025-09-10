package com.example.onnxinfer
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import ai.onnxruntime.*
import android.graphics.Canvas
import android.graphics.Color
import android.widget.ImageView
import java.nio.FloatBuffer
import java.nio.IntBuffer
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.android.OpenCVLoader
class MainActivity : AppCompatActivity() {

    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var detectorSession: OrtSession
    private lateinit var decoderSession: OrtSession

    private val textThreshold = 0.8f
    private val linkThreshold = 0.7f

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
            val bitmap = assets.open("easydemoupside.png").use {
                BitmapFactory.decodeStream(it)
            }

            Log.d(TAG, "Starting EasyOCR pipeline...")
            val startTime = System.currentTimeMillis()

            // Step 1: Preprocess for detector (EasyOCR style)
            val (detectorInput, ratios) = detectorPreprocess(bitmap)

            // Step 2: Run detector
            val detectorOutputs = runDetector(detectorInput)
            detectorInput.close()
            Log.e(TAG, "Found ${detectorOutputs.size()} text boxes")


            // Step 3: Post-process detector results to get text boxes
            val textBoxes = detectorPostprocess(detectorOutputs, ratios)
            detectorOutputs.forEach { it.value.close() }

            val boxedBitmap = drawBoxesOnBitmap(bitmap, textBoxes)

            runOnUiThread {
                val imageView = findViewById<ImageView>(R.id.resultImageView)
                imageView.setImageBitmap(boxedBitmap)
            }
            val totalTime = System.currentTimeMillis() - startTime
            Log.d(TAG, "=== EasyOCR RESULTS ===")
            Log.d(TAG, "Total time: ${totalTime}ms")
            Log.d(TAG, "Detected boxes: ${textBoxes.size}")

        } catch (e: Exception) {
            Log.e(TAG, "Complete EasyOCR pipeline failed: ${e.message}")
            e.printStackTrace()
        }
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
            val scoreOutput = outputs[0] as OnnxTensor
            val scores = scoreOutput.floatBuffer.array()
            Log.d("ONNX", "bruh = ${scores.size}")

            Log.e(TAG, "aaaaaaaaaaa = ${scoreOutput.info.shape}")

            val scoreData = scoreOutput.floatBuffer.array()

            val scoreShape = scoreOutput.info.shape

            Log.e(TAG, "${scoreShape.size}")

            val height = scoreShape?.get(1)?.toInt() ?: return emptyList()
            val width = scoreShape?.get(2)?.toInt() ?: return emptyList()
            val channels = scoreShape?.get(3)?.toInt() ?: return emptyList()



            Log.e(TAG, "h = ${height}")
            Log.e(TAG, "w = ${width}")
            Log.e(TAG, "c = $channels")

            if (channels >= 2) {
                // Extract text and link scores
                val textScores = Array(height) { FloatArray(width) }
                val linkScores = Array(height) { FloatArray(width) }

                for (y in 0 until height) {
                    for (x in 0 until width) {
                        val baseIdx = y * width * channels + x * channels
                        textScores[y][x] = scoreData[baseIdx]
                        linkScores[y][x] = scoreData[baseIdx + 1]
                        //Log.e(TAG, "TS = ${textScores[y][x]}")
                        //Log.e(TAG, "LS = ${linkScores[y][x]}")
                    }
                }


                // Apply adaptive thresholding


                // Create binary masks
                val textMask = createBinaryMask(textScores, 0.5f)
                val linkMask = createBinaryMask(linkScores, 0.3f)

                // Apply morphological operations to clean up the masks
                val cleanedTextMask = applyMorphologicalOperations(textMask)

                // Find connected components using proper algorithm
                val components = findConnectedComponents(cleanedTextMask, linkMask)

                // Filter and convert components to bounding boxes
                val filteredBoxes = filterAndConvertComponents(components, width, height)

                // Scale boxes back to original image coordinates
                textBoxes.addAll(filteredBoxes.map { box ->
                    TextBox(
                        (box.x1 * ratios.first).toInt(),
                        (box.y1 * ratios.second).toInt(),
                        (box.x2 * ratios.first).toInt(),
                        (box.y2 * ratios.second).toInt()
                    )
                })
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error in detector postprocess: ${e.message}")
            e.printStackTrace()
        }

        return textBoxes
    }


    // Data classes
    data class TextBox(val x1: Int, val y1: Int, val x2: Int, val y2: Int)






    override fun onDestroy() {
        super.onDestroy()
        if (::detectorSession.isInitialized) detectorSession.close()
        if (::decoderSession.isInitialized) decoderSession.close()
        if (::ortEnvironment.isInitialized) ortEnvironment.close()
    }



    private fun createBinaryMask(scores: Array<FloatArray>, threshold: Float): Array<BooleanArray> {
        val height = scores.size
        val width = scores[0].size
        return Array(height) { y ->
            BooleanArray(width) { x ->
                scores[y][x] > threshold
            }
        }
    }


    private fun applyMorphologicalOperations(mask: Array<BooleanArray>): Array<BooleanArray> {
        val height = mask.size
        val width = mask[0].size
        val result = Array(height) { BooleanArray(width) }

        // Apply opening (erosion followed by dilation) to remove noise
        val eroded = Array(height) { BooleanArray(width) }

        // Erosion
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                eroded[y][x] = mask[y][x] &&
                        mask[y - 1][x] && mask[y + 1][x] &&
                        mask[y][x - 1] && mask[y][x + 1]
            }
        }

        // Dilation
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                result[y][x] = eroded[y][x] ||
                        eroded[y - 1][x] || eroded[y + 1][x] ||
                        eroded[y][x - 1] || eroded[y][x + 1]
            }
        }

        return result
    }

    private fun findConnectedComponents(textMask: Array<BooleanArray>, linkMask: Array<BooleanArray>): List<List<Pair<Int, Int>>> {
        val height = textMask.size
        val width = textMask[0].size
        val visited = Array(height) { BooleanArray(width) }
        val components = mutableListOf<List<Pair<Int, Int>>>()

        for (y in 0 until height) {
            for (x in 0 until width) {
                if ((textMask[y][x] || linkMask[y][x]) && !visited[y][x]) {
                    val component = mutableListOf<Pair<Int, Int>>()
                    val queue = mutableListOf<Pair<Int, Int>>()
                    queue.add(Pair(x, y))
                    visited[y][x] = true

                    while (queue.isNotEmpty()) {
                        val (cx, cy) = queue.removeAt(0)
                        component.add(Pair(cx, cy))

                        // Check 8-connected neighbors
                        for (dy in -1..1) {
                            for (dx in -1..1) {
                                if (dx == 0 && dy == 0) continue
                                val nx = cx + dx
                                val ny = cy + dy

                                if (nx >= 0 && nx < width && ny >= 0 && ny < height && !visited[ny][nx]) {
                                    // Connect if it's a text pixel or if there's a link connection
                                    val isText = textMask[ny][nx]
                                    val hasLinkConnection = linkMask[ny][nx] &&
                                            (textMask[cy][cx] || linkMask[cy][cx])

                                    if (isText || hasLinkConnection) {
                                        visited[ny][nx] = true
                                        queue.add(Pair(nx, ny))
                                    }
                                }
                            }
                        }
                    }

                    if (component.isNotEmpty()) {
                        components.add(component)
                    }
                }
            }
        }

        return components
    }


    private fun filterAndConvertComponents(
        components: List<List<Pair<Int, Int>>>,
        imageWidth: Int,
        imageHeight: Int
    ): List<TextBox> {
        val boxes = mutableListOf<TextBox>()

        components.forEach { component ->
            if (component.size >= 10) { // Minimum component size
                val minX = component.minOf { it.first }
                val maxX = component.maxOf { it.first }
                val minY = component.minOf { it.second }
                val maxY = component.maxOf { it.second }

                val width = maxX - minX + 1
                val height = maxY - minY + 1
                val area = width * height
                val componentDensity = component.size.toFloat() / area

                // Filter based on reasonable text characteristics
                val aspectRatio = width.toFloat() / height
                val minDimension = minOf(width, height)
                val maxDimension = maxOf(width, height)

                boxes.add(TextBox(minX, minY, maxX + 1, maxY + 1))
                    Log.d(TAG, "Added filtered box: ($minX,$minY) to ($maxX,$maxY), " +
                            "size=${width}x${height}, density=${componentDensity}, ratio=${aspectRatio}")

            }
        }

        return boxes
    }





    private fun detectorPreprocess(bitmap: Bitmap): Pair<OnnxTensor, Pair<Float, Float>> {
        // Convert bitmap to RGB array
        val rgbArray = bitmapToRgbArray(bitmap)

        Log.d(TAG, "Original image size: ${bitmap.width}x${bitmap.height}")

        // Resize with proper aspect ratio preservation
        val (resizedImg, ratioW, ratioH) = resizeForDetection(rgbArray, canvasSize, magRatio)

        // Normalize using EasyOCR normalization
        val normalizedImg = normalizeMeanVariance(resizedImg)

        // Convert to tensor [1, 3, H, W]
        val height = normalizedImg.size
        val width = normalizedImg[0].size
        val channels = normalizedImg[0][0].size

        val floatArray = FloatArray(channels * height * width)

        // Transpose from HWC to CHW format
        for (c in 0 until channels) {
            for (h in 0 until height) {
                for (w in 0 until width) {
                    floatArray[c * height * width + h * width + w] = normalizedImg[h][w][c]
                }
            }
        }

        val shape = longArrayOf(1, channels.toLong(), height.toLong(), width.toLong())
        val tensor = OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(floatArray), shape)

        Log.d(TAG, "Preprocessed tensor shape: ${shape.contentToString()}")
        Log.d(TAG, "Resize ratios - W: $ratioW, H: $ratioH")

        return Pair(tensor, Pair(ratioW, ratioH))
    }

    private fun resizeForDetection(
        img: Array<Array<IntArray>>,
        canvasSize: Int,
        magRatio: Float
    ): Triple<Array<Array<FloatArray>>, Float, Float> {
        val originalHeight = img.size
        val originalWidth = img[0].size

        // Calculate target dimensions while preserving aspect ratio
        val targetSize = (canvasSize * magRatio).toInt()
        val ratio = 1f

        val newWidth = (originalWidth * ratio).toInt()
        val newHeight = (originalHeight * ratio).toInt()

        Log.d(TAG, "Calculated resize: ${originalWidth}x${originalHeight} -> ${newWidth}x${newHeight} (ratio: $ratio)")

        // Resize using bilinear interpolation
        val resized = bilinearResize(img, newWidth, newHeight)

        // Pad to target canvas size (800x608)
        val targetWidth = 800
        val targetHeight = 608
        val padded = Array(targetHeight) { Array(targetWidth) { FloatArray(3) } }

        // Center the image in the canvas
        val startX = (targetWidth - newWidth) / 2
        val startY = (targetHeight - newHeight) / 2

        // Fill with mean color (helps with normalization)
        val meanR = 127.5f
        val meanG = 127.5f
        val meanB = 127.5f

        for (y in 0 until targetHeight) {
            for (x in 0 until targetWidth) {
                if (y >= startY && y < startY + newHeight &&
                    x >= startX && x < startX + newWidth) {
                    val srcY = y - startY
                    val srcX = x - startX
                    padded[y][x][0] = resized[srcY][srcX][0]
                    padded[y][x][1] = resized[srcY][srcX][1]
                    padded[y][x][2] = resized[srcY][srcX][2]
                } else {
                    // Pad with mean values
                    padded[y][x][0] = meanR
                    padded[y][x][1] = meanG
                    padded[y][x][2] = meanB
                }
            }
        }

        // Calculate actual ratios for coordinate transformation
        val ratioW = originalWidth.toFloat() / targetWidth
        val ratioH = originalHeight.toFloat() / targetHeight

        return Triple(padded, ratioW, ratioH)
    }

    private fun bilinearResize(
        img: Array<Array<IntArray>>,
        newWidth: Int,
        newHeight: Int
    ): Array<Array<FloatArray>> {
        val originalHeight = img.size
        val originalWidth = img[0].size
        val resized = Array(newHeight) { Array(newWidth) { FloatArray(3) } }

        val xRatio = originalWidth.toFloat() / newWidth
        val yRatio = originalHeight.toFloat() / newHeight

        for (y in 0 until newHeight) {
            for (x in 0 until newWidth) {
                val px = x * xRatio
                val py = y * yRatio

                val x1 = px.toInt().coerceAtMost(originalWidth - 1)
                val y1 = py.toInt().coerceAtMost(originalHeight - 1)
                val x2 = (x1 + 1).coerceAtMost(originalWidth - 1)
                val y2 = (y1 + 1).coerceAtMost(originalHeight - 1)

                val dx = px - x1
                val dy = py - y1

                for (c in 0..2) {
                    val a = img[y1][x1][c] * (1 - dx) + img[y1][x2][c] * dx
                    val b = img[y2][x1][c] * (1 - dx) + img[y2][x2][c] * dx
                    resized[y][x][c] = (a * (1 - dy) + b * dy)
                }
            }
        }

        return resized
    }

    // Improved normalization with proper statistics
    private fun normalizeMeanVariance(img: Array<Array<FloatArray>>): Array<Array<FloatArray>> {
        val height = img.size
        val width = img[0].size
        val normalized = Array(height) { Array(width) { FloatArray(3) } }

        // EasyOCR uses ImageNet normalization, not (x - 127.5) / 127.5
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f) // ImageNet means
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)  // ImageNet stds

        for (y in 0 until height) {
            for (x in 0 until width) {
                // First normalize to [0, 1], then apply ImageNet normalization
                normalized[y][x][0] = (img[y][x][0] / 255.0f - mean[0]) / std[0]
                normalized[y][x][1] = (img[y][x][1] / 255.0f - mean[1]) / std[1]
                normalized[y][x][2] = (img[y][x][2] / 255.0f - mean[2]) / std[2]
            }
        }

        return normalized
    }
    private fun drawBoxesOnBitmap(bitmap: Bitmap, boxes: List<TextBox>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = android.graphics.Paint().apply {
            color = Color.BLUE
            style = android.graphics.Paint.Style.STROKE
            strokeWidth = 4f
            isAntiAlias = true
        }

        boxes.forEach { box ->
            canvas.drawRect(
                box.x1*2.toFloat(),
                box.y1*2.toFloat(),
                box.x2*2.toFloat(),
                box.y2*2.toFloat(),
                paint
            )
        }

        return mutableBitmap
    }











}