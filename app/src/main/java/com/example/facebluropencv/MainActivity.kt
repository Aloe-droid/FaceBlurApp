package com.example.facebluropencv

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.JavaCamera2View
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat

class MainActivity : ComponentActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    companion object {
        const val CAMERA_ID = 0
    }

    private val process by lazy { Process() }
    private var result: Mat? = null
    private var isDetect = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setPermissions()
        process.loadModel(assets, filesDir.toString())

        val openCVCameraView = ((JavaCamera2View(this, CAMERA_ID)) as CameraBridgeViewBase).apply {
            setCameraPermissionGranted()
            enableView()
            setCvCameraViewListener(this@MainActivity)
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }

        setContent {
            AndroidView(modifier = Modifier.fillMaxSize(), factory = { openCVCameraView })
        }
    }

    private fun setPermissions() {
        val requestPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestPermission()) {
                if (!it) {
                    Toast.makeText(this, "권한을 허용 하지 않으면 사용할 수 없습니다.", Toast.LENGTH_SHORT).show()
                    finish()
                }
            }

        listOf(Manifest.permission.CAMERA).forEach {
            if (ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED) {
                requestPermissionLauncher.launch(it)
            }
        }

        OpenCVLoader.initDebug()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
    }

    override fun onCameraViewStopped() {
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        val view = inputFrame!!.rgba()
        lifecycleScope.launch(Dispatchers.Default) {
            if(isDetect) return@launch

            isDetect = true
            result = process.detect(view)
            isDetect = false
        }

        return process.postProcess(view, result)
    }
}
