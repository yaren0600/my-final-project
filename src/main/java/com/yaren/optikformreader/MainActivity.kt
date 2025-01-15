package com.yaren.optikformreader

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.core.Scalar
import org.w3c.dom.Document
import java.io.File
import javax.xml.parsers.DocumentBuilderFactory
import javax.xml.transform.OutputKeys
import javax.xml.transform.TransformerFactory
import javax.xml.transform.dom.DOMSource
import javax.xml.transform.stream.StreamResult


class MainActivity : AppCompatActivity() {

    private val REQUEST_CODE_ANSWER_KEY = 1
    private val REQUEST_CODE_OPTICAL_FORM = 2
    private lateinit var processedImageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!OpenCVLoader.initDebug()) {
            showToast("OpenCV yüklenemedi! Uygulama kapanıyor.")
            finish() // OpenCV yüklenmezse uygulamayı kapat.
            return
        }

        processedImageView = findViewById(R.id.imageView_processed)

        findViewById<Button>(R.id.btn_answer_key).setOnClickListener {
            pickImageFromGallery(REQUEST_CODE_ANSWER_KEY)
        }

        findViewById<Button>(R.id.btn_optical_form).setOnClickListener {
            pickImageFromGallery(REQUEST_CODE_OPTICAL_FORM)
        }

        findViewById<Button>(R.id.btn_compare).setOnClickListener {
            compareAnswers()
        }
    }

    private fun pickImageFromGallery(requestCode: Int) {
        val intent = Intent(Intent.ACTION_PICK).apply {
            type = "image/*"
        }
        startActivityForResult(intent, requestCode)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == Activity.RESULT_OK && data?.data != null) {
            val imageUri = data.data!!

            when (requestCode) {
                REQUEST_CODE_ANSWER_KEY -> processAndSaveAnswers(imageUri, "answerkey.xml", "Cevap anahtarı kaydedildi.")
                REQUEST_CODE_OPTICAL_FORM -> processAndSaveAnswers(imageUri, "kullanici_cevap.xml", "Kullanıcı cevapları kaydedildi.")
            }
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
        val mat = Mat().apply { bitmapToMat(bitmap, this) }

        if (mat.empty()) {
            throw IllegalArgumentException("Yüklenen görüntü işlenemedi.")
        }

        val roiRect = Rect(0, 540, 1190, 1050)
        val roiMat = Mat(mat, roiRect)

        val processedMat = preprocessImage(roiMat)
        val boundingBoxes = findAnswerBoxes(processedMat)

        boundingBoxes.forEach { rect ->
            Imgproc.rectangle(roiMat, rect.tl(), rect.br(), Scalar(255.0, 0.0, 0.0), 2)
        }

        updateProcessedImageView(mat)
        return boundingBoxes.map { rect ->
            detectAnswer(rect, processedMat)
        }
    }


    private fun preprocessImage(mat: Mat): Mat {
        val grayMat = Mat().apply { Imgproc.cvtColor(mat, this, Imgproc.COLOR_BGR2GRAY) }
        saveImage(grayMat, "gray_image.jpg")

        val blurredMat = Mat().apply { Imgproc.GaussianBlur(grayMat, this, Size(3.0, 3.0), 0.0) }
        saveImage(blurredMat, "blurred_image.jpg")

        val thresholdMat = Mat()
        Imgproc.threshold(blurredMat, thresholdMat, 125.0, 255.0, Imgproc.THRESH_BINARY)
        saveImage(thresholdMat, "threshold_image.jpg")

        val cannyMat = Mat()
        Imgproc.Canny(blurredMat, cannyMat, 100.0, 200.0)
        saveImage(cannyMat, "canny_image.jpg")

        return thresholdMat

    }

    private fun findAnswerBoxes(thresholdMat: Mat): List<Rect> {
        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(thresholdMat, contours, Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)

        if (contours.isEmpty()) {
            throw IllegalArgumentException("Contours bulunamadı. Görüntü işleme başarısız.")
        }

        return contours.mapNotNull { contour ->
            val rect = Imgproc.boundingRect(contour)
            if (rect.height in 25..70 && rect.width in 25..70) rect else null
        }.sortedWith(compareBy({ it.y }, { it.x }))
    }
    private fun detectAnswer(rect: Rect, contourMat: Mat): String {
        val roi = Mat(contourMat, rect)
        val options = listOf("A", "B", "C", "D", "E") // Sorudaki şıklar
        val columnWidth = roi.cols() / options.size // Her şık için kolon genişliği

        // Her şık için, en ortadaki x koordinatına bakıyoruz.
        val xCenter = rect.x + rect.width / 2

        // Şıkların genişliği, her şık için kesişim noktaları oluşturuyor.
        val index = (xCenter / columnWidth).toInt()

        // Eğer index geçerli seçeneklerdeyse, doğru şık belirlenmiş olur.
        return if (index in options.indices) options[index] else "Boş" // Boş cevapları da kontrol et
    }



    private fun bitmapToMat(bitmap: Bitmap, mat: Mat) {
        org.opencv.android.Utils.bitmapToMat(bitmap.copy(Bitmap.Config.ARGB_8888, true), mat)
    }

    private fun updateProcessedImageView(mat: Mat) {
        saveImage(mat, "processed_image.jpg")

        val resultBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        org.opencv.android.Utils.matToBitmap(mat, resultBitmap)
        processedImageView.setImageBitmap(resultBitmap)
    }

    private fun saveImage(mat: Mat, fileName: String) {
        if (mat.empty()) {
            showToast("Görüntü kaydedilemedi çünkü matris boş.")
            return
        }

        val directory = File(getExternalFilesDir(null), "processed_images")
        if (!directory.exists()) {
            directory.mkdirs()
        }

        val filePath = File(directory, fileName).absolutePath
        if (org.opencv.imgcodecs.Imgcodecs.imwrite(filePath, mat)) {
            showToast("Görüntü kaydedildi: $fileName $filePath")
        } else {
            showToast("Görüntü kaydedilemedi: $filePath")
        }
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

    private fun loadAnswersFromXml(fileName: String): List<String> {
        val file = File(filesDir, fileName)
        if (!file.exists()) return emptyList()

        val document = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(file)
        return document.getElementsByTagName("Question").let { nodeList ->
            List(nodeList.length) { nodeList.item(it).textContent }
        }
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}
