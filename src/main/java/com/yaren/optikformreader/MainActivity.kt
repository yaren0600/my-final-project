package com.yaren.optikformreader

import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.w3c.dom.Document
import java.io.File
import javax.xml.parsers.DocumentBuilderFactory
import javax.xml.transform.OutputKeys
import javax.xml.transform.TransformerFactory
import javax.xml.transform.dom.DOMSource
import javax.xml.transform.stream.StreamResult

class MainActivity : AppCompatActivity() {

    private lateinit var processedImageView: ImageView

    private val getAnswerKeyImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { processAndSaveAnswers(it, "answerkey.xml", "Cevap anahtarı kaydedildi.") }
    }

    private val getOpticalFormImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { processAndSaveAnswers(it, "kullanici_cevap.xml", "Kullanıcı cevapları kaydedildi.") }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!OpenCVLoader.initDebug()) {
            showToast("OpenCV yüklenemedi! Uygulama kapanıyor.")
            finish()
            return
        }

        processedImageView = findViewById(R.id.imageView_processed)

        findViewById<Button>(R.id.btn_answer_key).setOnClickListener {
            getAnswerKeyImage.launch("image/*")
        }

        findViewById<Button>(R.id.btn_optical_form).setOnClickListener {
            getOpticalFormImage.launch("image/*")
        }

        findViewById<Button>(R.id.btn_compare).setOnClickListener {
            compareAnswers()
        }
    }

    private fun processAndSaveAnswers(imageUri: Uri, fileName: String, successMessage: String) {
        try {
            val answers = processOpticalForm(imageUri)
            saveAnswersToXml(fileName, answers)
            showToast(successMessage)
        } catch (e: Exception) {
            showToast("Bir hata oluştu: ${e.message}")
        }
    }

    private fun processOpticalForm(imageUri: Uri): List<String> {
        val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, imageUri)
        val mat = Mat()
        bitmapToMat(bitmap, mat)

        if (mat.empty()) throw IllegalArgumentException("Yüklenen görüntü işlenemedi.")

        // Görüntü işleme: Gri tonlama, bulanıklaştırma ve kenar tespiti
        val processedMat = preprocessImage(mat)

        // Dikdörtgenleri buluyoruz
        val boundingBoxes = findRectangles(processedMat)

        // Dikdörtgenleri işaretliyoruz
        boundingBoxes.forEachIndexed { index, rect ->
            Imgproc.rectangle(mat, rect.tl(), rect.br(), Scalar(255.0, 0.0, 0.0), 2) // Kırmızı dikdörtgen

            // Numara yazma
            val text = (index + 1).toString()
            val font = Imgproc.FONT_HERSHEY_SIMPLEX
            val fontScale = 1.0
            val color = Scalar(255.0, 255.0, 255.0) // Beyaz renk
            val thickness = 2
            val baseline = IntArray(1)
            val textSize = Imgproc.getTextSize(text, font, fontScale, thickness, baseline)

            val textX = rect.x + (rect.width / 2) - (textSize.width / 2).toInt()
            val textY = rect.y - 10 // Numara dikdörtgenin biraz üstünde olacak

            // Numara yazma
            Imgproc.putText(mat, text, Point(textX.toDouble(), textY.toDouble()), font, fontScale, color, thickness)
        }

        updateProcessedImageView(mat) // Ekranda güncel görüntüyü göster

        // Cevapları tespit ediyoruz
        return boundingBoxes.map { rect -> detectAnswers(rect, processedMat) }
    }

    private fun preprocessImage(mat: Mat): Mat {
        // Gri tonlama
        val grayMat = Mat()
        Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_BGR2GRAY)

        // Histogram eşitleme
        val equalizedMat = Mat()
        Imgproc.equalizeHist(grayMat, equalizedMat)

        // Gaussian Blur
        val blurredMat = Mat()
        Imgproc.GaussianBlur(equalizedMat, blurredMat, Size(5.0, 5.0), 0.0)

        // Kenar tespiti (Canny)
        val edgesMat = Mat()
        Imgproc.Canny(blurredMat, edgesMat, 100.0, 200.0)

        return edgesMat
    }

    private fun findRectangles(edgesMat: Mat): List<Rect> {
        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(
            edgesMat,
            contours,
            Mat(),
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        if (contours.isEmpty()) throw IllegalArgumentException("Contours bulunamadı. Görüntü işleme başarısız.")

        // Yalnızca dikdörtgen olan konturları buluyoruz
        val rectangles = mutableListOf<Rect>()
        contours.forEach { contour ->
            val approx = MatOfPoint2f()
            val contour2f = MatOfPoint2f(*contour.toArray())
            val epsilon = Imgproc.arcLength(contour2f, true) * 0.02
            Imgproc.approxPolyDP(contour2f, approx, epsilon, true)

            if (approx.total() == 4L) {
                val rect = Imgproc.boundingRect(approx)
                if (rect.width > 50 && rect.height > 50) {
                    rectangles.add(rect)
                }
            }
        }
        return rectangles
    }

    private fun detectAnswers(boundingBox: Rect, contourMat: Mat): String {
        val options = listOf("A", "B", "C", "D", "E")
        val roi = Mat(contourMat, boundingBox)
        val columnWidth = roi.cols() / options.size

        // Her bir seçenek için işaretleme oranı hesaplanır
        val fillPercentages = options.indices.map { index ->
            val startX = index * columnWidth
            val endX = startX + columnWidth
            val columnRoi = roi.colRange(startX, endX)

            // Non-zero piksel sayısını al
            val nonZeroCount = Core.countNonZero(columnRoi)
            val totalPixels = columnRoi.rows() * columnRoi.cols()
            nonZeroCount.toDouble() / totalPixels * 100
        }

        // Eşik değerini daha düşük yapıyoruz. (30% yerine 10%)
        val thresholdPercentage = 10.0 // %10'unu geçen işaretleme oranı, işaretli kabul edilir

        val maxPercentage = fillPercentages.maxOrNull() ?: 0.0
        val maxIndex = fillPercentages.indexOf(maxPercentage)

        return if (maxPercentage > thresholdPercentage && maxIndex in options.indices) {
            options[maxIndex]
        } else {
            "Boş"
        }
    }

    private fun bitmapToMat(bitmap: Bitmap, mat: Mat) {
        org.opencv.android.Utils.bitmapToMat(bitmap.copy(Bitmap.Config.ARGB_8888, true), mat)
    }

    private fun updateProcessedImageView(mat: Mat) {
        val resultBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        org.opencv.android.Utils.matToBitmap(mat, resultBitmap)
        processedImageView.setImageBitmap(resultBitmap)
    }

    private fun saveAnswersToXml(fileName: String, answers: List<String>) {
        val file = File(filesDir, fileName)
        val documentBuilder = DocumentBuilderFactory.newInstance().newDocumentBuilder()
        val document: Document = documentBuilder.newDocument()

        val rootElement = document.createElement("Answers")
        document.appendChild(rootElement)

        answers.forEachIndexed { index, answer ->
            rootElement.appendChild(document.createElement("Question").apply {
                setAttribute("number", (index + 1).toString())
                textContent = answer
            })
        }

        TransformerFactory.newInstance().newTransformer().apply {
            setOutputProperty(OutputKeys.INDENT, "yes")
            setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "2")
            transform(DOMSource(document), StreamResult(file))
        }
    }

    private fun loadAnswersFromXml(fileName: String): List<String> {
        val file = File(filesDir, fileName)
        if (!file.exists()) return emptyList()

        val document = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(file)
        return document.getElementsByTagName("Question").let { nodeList ->
            List(nodeList.length) { nodeList.item(it).textContent }
        }
    }

    private fun compareAnswers() {
        val answerKey = loadAnswersFromXml("answerkey.xml")
        val userAnswers = loadAnswersFromXml("kullanici_cevap.xml")

        val (correct, wrong, empty) = answerKey.indices.fold(Triple(0, 0, 0)) { acc, i ->
            when {
                i >= userAnswers.size || userAnswers[i].isEmpty() -> acc.copy(third = acc.third + 1)
                answerKey[i] == userAnswers[i] -> acc.copy(first = acc.first + 1)
                else -> acc.copy(second = acc.second + 1)
            }
        }

        showToast("Doğru: $correct, Yanlış: $wrong, Boş: $empty")
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}
