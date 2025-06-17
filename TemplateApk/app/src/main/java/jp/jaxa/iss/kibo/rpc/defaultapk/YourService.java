package jp.jaxa.iss.kibo.rpc.defaultapk; // Use your team's application ID

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.opencv.aruco.Aruco;
import org.opencv.aruco.Dictionary;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.task.vision.detector.Detection;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import gov.nasa.arc.astrobee.Result;
import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

public class YourService extends KiboRpcService {

    // TODO: IMPORTANT! Change this to the exact name of your model file in the assets folder.
    private final String TFLITE_MODEL_NAME = "model.tflite";

    private YOLOv8Detector yoloDetector;
    private final Map<Integer, Point> patrolPoints = new HashMap<>();
    private final Map<Integer, Quaternion> patrolOrientations = new HashMap<>();
    private final Map<String, Object> missionData = new HashMap<>();

    @Override
    protected void runPlan1() {
        initializeMissionParameters();

        try {
            yoloDetector = new YOLOv8Detector(getApplicationContext(), TFLITE_MODEL_NAME);
            Log.i("MISSION_PHASE", "YOLOv8 Detector initialized successfully.");
        } catch (IOException e) {
            Log.e("CRITICAL_ERROR", "Failed to initialize YOLOv8 detector. Aborting.", e);
            return;
        }

        api.startMission();

        // Phase 1: Patrol all areas using a defined strategy
        patrolAndDetect(1);
        patrolAndDetect(2);
        patrolAndDetect(3);
        patrolAndDetect(4);

        // Report all findings to the system
        for(int area = 1; area <= 4; area++) {
            api.setAreaInfo(area, (String) missionData.getOrDefault("Area" + area + "_Item", "none"), (Integer) missionData.getOrDefault("Area" + area + "_Count", 0));
        }

        // Phase 2: Report to Astronaut and get the final clue
        moveToWrapper(patrolPoints.get(10), patrolOrientations.get(10));
        api.reportRoundingCompletion(); //

        ItemDetectionResult targetClue = patrolAndDetect(5); // Area 5 is the astronaut's clue
        // TODO: The rulebook says the astronaut shows one Treasure and two Landmark items.
        // You must implement logic here to determine which of the detected items is the *actual* target.
        // For this template, we assume the first detected item is the target.
        missionData.put("FinalTargetItem", targetClue.getItemName());
        Log.i("MISSION_PHASE", "Final Target identified as: " + targetClue.getItemName());
        api.notifyRecognitionItem(); //

        // Phase 3: Proceed to the final target and complete the mission
        String finalTargetItem = (String) missionData.get("FinalTargetItem");
        int finalArea = 0;
        for (int area = 1; area <= 4; area++) {
            if (Objects.equals(missionData.get("Area" + area + "_Item"), finalTargetItem)) {
                finalArea = area;
                break;
            }
        }

        if (finalArea != 0) {
            Log.i("MISSION_PHASE", "Final target '" + finalTargetItem + "' is in Area " + finalArea + ". Moving to take snapshot.");
            moveToWrapper(patrolPoints.get(finalArea), patrolOrientations.get(finalArea));
            // Perform a final, precise turn for a perfect shot
            List<Double> finalTurnAngles = calculateFinalTurn(finalArea);
            moveToWrapper(patrolPoints.get(finalArea), QuaternionUtils.computeQuaternionFromAngles(finalTurnAngles));
        } else {
            Log.e("MISSION_ERROR", "Could not locate final target '" + finalTargetItem + "'. Taking snapshot at a default location.");
            moveToWrapper(patrolPoints.get(1), patrolOrientations.get(1)); // Fallback
        }
        api.takeTargetItemSnapshot(); //
    }

    private ItemDetectionResult patrolAndDetect(int areaNumber) {
        Log.i("PATROL_LOGIC", "Processing Area: " + areaNumber);
        if (areaNumber <= 4) { // Standard patrol
            moveToWrapper(patrolPoints.get(areaNumber), patrolOrientations.get(areaNumber));
        }
        // Area 5 (Astronaut) does not require movement as we are already there.

        String imagePath = AR_cropping(areaNumber);
        if (imagePath == null) {
            Log.e("AR_ERROR", "AR Cropping failed for area " + areaNumber);
            return new ItemDetectionResult("error", 0);
        }

        return predictItemsInArea(imagePath, areaNumber);
    }

    private ItemDetectionResult predictItemsInArea(String imagePath, int areaNumber) {
        Bitmap bitmap = loadBitmapFromSDCard(imagePath);
        if (bitmap == null) {
            Log.e("YOLOv8_ERROR", "Could not load cropped image: " + imagePath);
            return new ItemDetectionResult("error", 0);
        }

        List<Detection> detections = yoloDetector.detect(bitmap);
        if (detections == null || detections.isEmpty()) {
            return new ItemDetectionResult("none", 0);
        }

        // TODO: Implement your own logic to count and filter detections.
        // This could involve Non-Maximum Suppression or other advanced techniques.
        // For now, we assume the most confident detection is the item type and count all detections.
        String mostLikelyItem = detections.get(0).getCategories().get(0).getLabel();
        int itemCount = detections.size();
        return new ItemDetectionResult(mostLikelyItem, itemCount);
    }

    private void initializeMissionParameters() {
        // Points for patrolling areas. Fine-tune these values in the simulator.
        // Area Coordinates from Rulebook
        patrolPoints.put(1, new Point(10.9, -9.8, 5.0));
        patrolOrientations.put(1, new Quaternion(0f, 0.707f, 0f, 0.707f));
        patrolPoints.put(2, new Point(10.9, -8.8, 4.5));
        patrolOrientations.put(2, new Quaternion(0.5f, 0.5f, -0.5f, 0.5f));
        patrolPoints.put(3, new Point(10.9, -7.9, 4.5));
        patrolOrientations.put(3, new Quaternion(0.5f, 0.5f, -0.5f, 0.5f));
        patrolPoints.put(4, new Point(10.6, -7.0, 5.0));
        patrolOrientations.put(4, new Quaternion(0f, 0f, -0.707f, 0.707f));
        // Astronaut location from Rulebook
        patrolPoints.put(10, new Point(11.143, -6.7607, 4.9654));
        patrolOrientations.put(10, new Quaternion(0f, 0f, 0.707f, 0.707f));
    }

    private boolean moveToWrapper(Point point, Quaternion quaternion) {
        Result result = api.moveTo(point, quaternion, true);
        int loopCounter = 0;
        while (!result.hasSucceeded() && loopCounter < 3) {
            result = api.moveTo(point, quaternion, true);
            loopCounter++;
        }
        return result.hasSucceeded();
    }

    private String AR_cropping(int targetNum) {
        // Use NavCam for areas 1-4, DockCam for astronaut's clue (area 5)
        Mat image = (targetNum == 5) ? api.getMatDockCam() : api.getMatNavCam();
        double[][] cameraIntrinsics = (targetNum == 5) ? api.getDockCamIntrinsics() : api.getNavCamIntrinsics();
        if (image == null) return null;

        Mat cameraMatrix = new Mat(3, 3, org.opencv.core.CvType.CV_32FC1);
        cameraMatrix.put(0, 0, cameraIntrinsics[0]);
        Mat distCoeffs = new Mat(1, 5, org.opencv.core.CvType.CV_32FC1);
        distCoeffs.put(0, 0, cameraIntrinsics[1]);

        Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat();
        Aruco.detectMarkers(image, dictionary, corners, ids);

        if (ids.empty()) return null;

        // Simplified logic: use the first detected marker to define the crop area
        MatOfPoint2f cornerPoints = new MatOfPoint2f(corners.get(0));
        Rect roi = Imgproc.boundingRect(cornerPoints);
        // TODO: The champion's code has much more advanced logic to calculate the precise
        // crop area based on marker pose. This simplified version is a starting point.
        Mat cropped = new Mat(image, roi);
        String imagePath = "sdcard/data/" + getApplicationContext().getPackageName() + "/immediate/DebugImages/post_" + targetNum + ".png";
        api.saveMatImage(cropped, "post_" + targetNum + ".png");
        return imagePath;
    }

    private List<Double> calculateFinalTurn(int targetNum) {
        // TODO: This method should be implemented based on the champion's `Final_turn` logic.
        // It detects the AR tag again and calculates the roll/pitch/yaw deviation from the
        // camera center to perform a highly accurate final aim.
        List<Double> angles = new ArrayList<>();
        angles.add(0.0); // Yaw
        angles.add(0.0); // Pitch
        angles.add(0.0); // Roll
        return angles;
    }

    public static Bitmap loadBitmapFromSDCard(String filePath) {
        File imgFile = new File(filePath);
        return imgFile.exists() ? BitmapFactory.decodeFile(imgFile.getAbsolutePath()) : null;
    }

    private static class ItemDetectionResult {
        private final String itemName;
        private final int itemCount;
        public ItemDetectionResult(String name, int count) { this.itemName = name; this.itemCount = count; }
        public String getItemName() { return itemName; }
        public int getItemCount() { return itemCount; }
    }
}