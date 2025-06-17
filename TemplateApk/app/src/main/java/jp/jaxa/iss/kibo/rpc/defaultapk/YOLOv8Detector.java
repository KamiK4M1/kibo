package jp.jaxa.iss.kibo.rpc.defaultapk;

import android.content.Context;
import android.graphics.Bitmap;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.core.BaseOptions;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

public class YOLOv8Detector {
    private ObjectDetector objectDetector;

    public YOLOv8Detector(Context context, String modelName) throws IOException {
        File modelFile = convertAssetToFile(context, modelName);
        ObjectDetector.ObjectDetectorOptions options =
                ObjectDetector.ObjectDetectorOptions.builder()
                        .setBaseOptions(BaseOptions.builder().build())
                        .setScoreThreshold(0.5f) // Adjust this threshold
                        .setMaxResults(5)
                        .build();
        objectDetector = ObjectDetector.createFromFileAndOptions(modelFile, options);
    }

    public List<Detection> detect(Bitmap bitmap) {
        if (objectDetector == null || bitmap == null) return null;
        return objectDetector.detect(TensorImage.fromBitmap(bitmap));
    }

    private File convertAssetToFile(Context context, String modelFileName) throws IOException {
        InputStream inputStream = context.getAssets().open(modelFileName);
        File tempFile = File.createTempFile("model", ".tflite");
        tempFile.deleteOnExit();
        try (FileOutputStream outputStream = new FileOutputStream(tempFile)) {
            byte[] buffer = new byte[4 * 1024];
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, read);
            }
        }
        return tempFile;
    }
}