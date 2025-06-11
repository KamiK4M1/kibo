package jp.jaxa.iss.kibo.rpc.Aht_Ja_Yung;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.Utils;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import gov.nasa.arc.astrobee.Kinematics;
import gov.nasa.arc.astrobee.Result;
import gov.nasa.arc.astrobee.types.Quaternion;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import jp.jaxa.iss.kibo.rpc.sampleapk.R;

/**
 * Your main service class for the Kibo Robot Programming Challenge.
 * This class contains the logic to control the Astrobee robot.
 */
public class YourService extends KiboRpcService {

    // A map to store the identified items in each area
    private Map<Integer, String> areaItemMap = new HashMap<>();
    private Map<String, Mat> templates = new HashMap<>();
    private Context context;
    private boolean templatesLoaded = false;
    // List of treasure items to distinguish them from landmark items
    private final List<String> TREASURE_ITEMS = Arrays.asList("crystal", "diamond", "emerald");
    private final List<String> LANDMARK_ITEMS = Arrays.asList("coral", "coin", "compass", "fossil", "key", "letter", "shell", "treasure_box");


    @Override
    public void onCreate() {
        super.onCreate();
        // Get the application context for loading resources
        this.context = getApplicationContext();
        loadAllTemplates();
    }

    /**
     * This is the main mission logic that will be executed for the competition.
     */
    @Override
    protected void runPlan1() {
        // 1. Start the mission
        api.startMission();

        // 2. Define coordinates for all 4 areas
        Map<Integer, gov.nasa.arc.astrobee.types.Point> areaDestinations = new HashMap<>();
        areaDestinations.put(1, new gov.nasa.arc.astrobee.types.Point(10.95d, -10.58d, 5.195d));
        areaDestinations.put(2, new gov.nasa.arc.astrobee.types.Point(10.925d, -8.875d, 3.76203d));
        areaDestinations.put(3, new gov.nasa.arc.astrobee.types.Point(10.925d, -7.925d, 3.76093d));
        areaDestinations.put(4, new gov.nasa.arc.astrobee.types.Point(9.866984d, -6.8525d, 4.945d));
        Quaternion areaOrientation = new Quaternion(0f, 0f, 0f, 1f); // EXAMPLE: Adjust per-area.

        // 3. Loop through each area to patrol and identify items
        for (int areaId = 1; areaId <= 4; areaId++) {
            moveToWrapper(areaDestinations.get(areaId), areaOrientation);
            Mat image = api.getMatNavCam();

            if (image != null) {
                api.saveMatImage(image, "area_" + areaId + "_raw.png");

                // --- Call the OpenCV image recognition function ---
                Map<String, Integer> foundItems = recognizeItems(image);

                // --- Process the results ---
                String bestItem = "unknown";
                int bestCount = 0;
                // This is a simple logic to find the most frequent item, you might need something more complex.
                for(Map.Entry<String, Integer> entry : foundItems.entrySet()){
                    if(entry.getValue() > bestCount){
                        bestItem = entry.getKey();
                        bestCount = entry.getValue();
                    }
                }

                areaItemMap.put(areaId, bestItem);
                api.setAreaInfo(areaId, bestItem, bestCount);
            }
        }

        // 4. Move to the astronaut's location
        gov.nasa.arc.astrobee.types.Point astronautPoint = new gov.nasa.arc.astrobee.types.Point(11.143d, -6.7607d, 4.9654d);
        Quaternion astronautQuaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);
        moveToWrapper(astronautPoint, astronautQuaternion);
        api.reportRoundingCompletion();

        // 5. Get the image of the target item
        Mat targetImage = api.getMatNavCam();
        String realTreasureName = "";
        if (targetImage != null) {
            // --- Call OpenCV logic again for the target image ---
            Map<String, Integer> targetContents = recognizeItems(targetImage);
            // Loop through the items found in the astronaut's image
            // and identify which one is a treasure.
            for (String itemName : targetContents.keySet()) {
                if (TREASURE_ITEMS.contains(itemName)) {
                    realTreasureName = itemName;
                    break; // Found the treasure, no need to look further
                }
            }
        }
        api.notifyRecognitionItem();

        // 6. Find which area the real treasure was in
        int treasureAreaId = -1;
        for (Map.Entry<Integer, String> entry : areaItemMap.entrySet()) {
            if (entry.getValue().equals(realTreasureName)) {
                treasureAreaId = entry.getKey();
                break;
            }
        }

        // 7. Move to the treasure's location
        if (treasureAreaId != -1) {
            moveToWrapper(areaDestinations.get(treasureAreaId), areaOrientation);
        }

        // 8. Take the final snapshot
        api.takeTargetItemSnapshot();
    }

    /**
     * A wrapper function for the moveTo API call.
     * FIX: This function now waits for the robot to physically arrive at the destination
     * before returning, to prevent getting the same camera image multiple times.
     */
    private void moveToWrapper(gov.nasa.arc.astrobee.types.Point targetPoint, Quaternion quaternion) {
        final int MAX_RETRIES = 3;
        int retryCount = 0;
        Result result = null;

        do {
            result = api.moveTo(targetPoint, quaternion, true);
            retryCount++;
        } while (!result.hasSucceeded() && retryCount < MAX_RETRIES);

        // If the command was successfully sent, wait for the robot to arrive.
        if (result.hasSucceeded()) {
            Kinematics kinematics;
            final double ARRIVAL_THRESHOLD = 0.05; // 5 cm threshold
            long startTime = System.currentTimeMillis();
            long timeout = 30000; // 30 second timeout

            while (true) {
                // Check for timeout
                if (System.currentTimeMillis() - startTime > timeout) {
                    Log.e("KiboRPC", "Timeout waiting for robot to arrive at destination.");
                    break; // Exit the waiting loop
                }

                kinematics = api.getRobotKinematics();
                double distance = calculateDistance(kinematics.getPosition(), targetPoint);

                // If we are close enough, break the loop
                if (distance < ARRIVAL_THRESHOLD) {
                    break;
                }

                try {
                    // Wait a short amount of time before checking again
                    Thread.sleep(250);
                } catch (InterruptedException e) {
                    Log.e("KiboRPC", "Thread sleep interrupted", e);
                }
            }
        }
    }

    /**
     * Helper function to calculate the Euclidean distance between two points.
     * @param p1 The first point.
     * @param p2 The second point.
     * @return The distance between the two points.
     */
    private double calculateDistance(gov.nasa.arc.astrobee.types.Point p1, gov.nasa.arc.astrobee.types.Point p2) {
        double dx = p1.getX() - p2.getX();
        double dy = p1.getY() - p2.getY();
        double dz = p1.getZ() - p2.getZ();
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    /**
     * Loads all template images from the drawable resources into memory.
     */
    private void loadAllTemplates() {
        if (templatesLoaded) return;
        templates.put("coral", loadTemplate(R.drawable.coral));
        templates.put("coin", loadTemplate(R.drawable.coin));
        templates.put("compass", loadTemplate(R.drawable.compass));
        templates.put("fossil", loadTemplate(R.drawable.fossil));
        templates.put("key", loadTemplate(R.drawable.key));
        templates.put("letter", loadTemplate(R.drawable.letter));
        templates.put("shell", loadTemplate(R.drawable.shell));
        templates.put("treasure_box", loadTemplate(R.drawable.treasure_box));
        templates.put("crystal", loadTemplate(R.drawable.crystal));
        templates.put("diamond", loadTemplate(R.drawable.diamond));
        templates.put("emerald", loadTemplate(R.drawable.emerald));
        templatesLoaded = true;
    }

    /**
     * Helper function to load a single template image from resources and convert to an OpenCV Mat.
     */
    private Mat loadTemplate(int resourceId) {
        try {
            Bitmap bmp = BitmapFactory.decodeResource(context.getResources(), resourceId);
            Mat mat = new Mat();
            Utils.bitmapToMat(bmp, mat);
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY);
            return mat;
        } catch (Exception e) {
            Log.e("KiboRPC", "Error loading template: " + e.getMessage());
            return new Mat();
        }
    }

    /**
     * Analyzes the camera image to find and count items using the loaded templates.
     */
    private Map<String, Integer> recognizeItems(Mat cameraImage) {
        Map<String, Integer> foundItems = new HashMap<>();
        Mat grayCameraImage = new Mat();
        Imgproc.cvtColor(cameraImage, grayCameraImage, Imgproc.COLOR_BGR2GRAY);

        for (Map.Entry<String, Mat> templateEntry : templates.entrySet()) {
            String itemName = templateEntry.getKey();
            Mat template = templateEntry.getValue();
            if (template.empty()) continue;

            Mat result = new Mat();
            Imgproc.matchTemplate(grayCameraImage, template, result, Imgproc.TM_CCOEFF_NORMED);
            double threshold = 0.8;

            while (true) {
                Core.MinMaxLocResult mmr = Core.minMaxLoc(result);
                Point matchLoc = mmr.maxLoc;
                if (mmr.maxVal >= threshold) {
                    foundItems.put(itemName, foundItems.getOrDefault(itemName, 0) + 1);
                    Imgproc.rectangle(cameraImage, matchLoc, new Point(matchLoc.x + template.cols(), matchLoc.y + template.rows()), new Scalar(0, 255, 0), 3);
                    Imgproc.rectangle(result, matchLoc, new Point(matchLoc.x + template.cols(), matchLoc.y + template.rows()), new Scalar(0), -1);
                } else {
                    break;
                }
            }
        }
        api.saveMatImage(cameraImage, "processed_image_" + System.currentTimeMillis() + ".png");
        return foundItems;
    }

    @Override
    protected void runPlan2() { /* Your code here */ }

    @Override
    protected void runPlan3() { /* Your code here */ }
}
