package jp.jaxa.iss.kibo.rpc.defaultapk;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import org.tensorflow.lite.Interpreter;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

// This class will handle all TensorFlow Lite model operations.
public class ObjectDetector {

    private Interpreter interpreter;
    private List<String> labelList;
    private int INPUT_SIZE;
    private int PIXEL_SIZE = 3; // For RGB
    private int IMAGE_MEAN = 0;
    private float IMAGE_STD = 255.0f;

    // A helper class to store detection results
    public static class DetectionResult {
        public final String label;
        public final float score;
        public final android.graphics.RectF boundingBox;

        public DetectionResult(String label, float score, android.graphics.RectF boundingBox) {
            this.label = label;
            this.score = score;
            this.boundingBox = boundingBox;
        }
    }

    public static ObjectDetector create(Context context, final String modelPath, final String labelPath, int inputSize) throws IOException {
        ObjectDetector objectDetector = new ObjectDetector();
        objectDetector.interpreter = new Interpreter(objectDetector.loadModelFile(context.getAssets(), modelPath));
        objectDetector.labelList = objectDetector.loadLabelList(context.getAssets(), labelPath);
        objectDetector.INPUT_SIZE = inputSize;
        return objectDetector;
    }

    // Loads the TFLite model from the assets folder.
    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // Loads the labels from the assets folder.
    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    // Main detection method
    public List<DetectionResult> detect(Bitmap bitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(resizedBitmap);

        // Output format for YOLOv8 is typically [1][84][8400]
        // Bounding box, class scores
        // We need to transpose this to [1][8400][84]
        Object[] inputArray = {byteBuffer};
        float[][][] output = new float[1][8400][labelList.size() + 4]; // [batch][predictions][box+classes]

        java.util.Map<Integer, Object> outputMap = new java.util.HashMap<>();
        outputMap.put(0, output);

        interpreter.runForMultipleInputsOutputs(inputArray, outputMap);

        // Process the output and apply Non-Max Suppression
        return processOutput(output[0]);
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat(((val & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }
        return byteBuffer;
    }

    // Process YOLOv8 output and perform NMS
    private List<DetectionResult> processOutput(float[][] output) {
        // Use a PriorityQueue to keep track of the top N results.
        final int MAX_RESULTS = 10; // Max number of detections to return
        PriorityQueue<DetectionResult> pq = new PriorityQueue<>(
                MAX_RESULTS,
                new java.util.Comparator<DetectionResult>() {
                    @Override
                    public int compare(DetectionResult lhs, DetectionResult rhs) {
                        // Sort by score in descending order
                        return Float.compare(rhs.score, lhs.score);
                    }
                }
        );

        // Transpose and process detections
        for (int i = 0; i < output.length; i++) {
            float confidence = 0.0f;
            int detectedClass = -1;

            // Find the class with the highest score
            for (int c = 4; c < output[i].length; c++) {
                if (output[i][c] > confidence) {
                    confidence = output[i][c];
                    detectedClass = c - 4;
                }
            }

            // Filter by confidence threshold
            float CONFIDENCE_THRESHOLD = 0.5f;
            if (confidence > CONFIDENCE_THRESHOLD && detectedClass < labelList.size()) {
                float cx = output[i][0];
                float cy = output[i][1];
                float w = output[i][2];
                float h = output[i][3];
                float left = cx - w / 2;
                float top = cy - h / 2;
                float right = cx + w / 2;
                float bottom = cy + h / 2;

                android.graphics.RectF boundingBox = new android.graphics.RectF(left, top, right, bottom);
                String label = labelList.get(detectedClass);
                pq.add(new DetectionResult(label, confidence, boundingBox));
            }
        }

        // You should apply Non-Maximum Suppression (NMS) here.
        // For simplicity, we are returning top results, but for real competition, NMS is crucial.
        // You can adapt the NonMaxSuppression.java class from the 5th kibo champion for this.

        final ArrayList<DetectionResult> detections = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            detections.add(pq.poll());
        }
        return detections;
    }
}