package edu.onbasli.indoorlocalization.InertiaNavegation.floorDetection;

import android.app.Activity;
import android.content.pm.PackageManager;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;

import java.io.IOException;
import java.util.Vector;

import edu.onbasli.indoorlocalization.InertiaNavegation.filewriting.DataFileWriter;

public class FloorDetection {

    private static final String FOLDER_NAME = "Pedestrian_Dead_Reckoning/Floor_detect";
    private static final String[] DATA_FILE_NAMES = {
            "Floor_Detect"
    };
    private static final String[] DATA_FILE_HEADINGS = {
            "Barometer,currentAvg,AvgBefore_2_Second,Pstart,Pend,CurrentFloor"
    };
    private DataFileWriter dataFileWriter;

    //height estimation
    // private final int timeToUpdate = 15; //time which after we will see if the floor number changed
    private final int timeToAverage = 5; //Take avg readings every 1s
    private Vector<Float> avgReadings;
    private float avgT_2 = 0; //average values before 3 seconds;
    private float avgT_0 = 0; //average values in the last 1 seconds;
    private float avgT_5 = 0; //average values in the last 5 seconds;
    private float pstart = 0; //p at start walking up or down;
    private float pend = 0; //p at the end of walking up or down;

    private int currentFloor = 0;
    // private float lastAverageTime, lastUpdateTime = 0;
    private final float mPh_btwFloors = 0.30f;
    private boolean first_run = true;
    private boolean firstRun = true;
    private TextView floorNum;
    private float currentTime = 0;
    private final float thetaT = 0.01f; //hpa
    private int N0 , N1 = 5; //to remove ping-pong effect
    private int num1, num2 = 0; // counters
    private boolean enterLoop2 = false;
    private int upORdown = 0; //1 means up 2 means down
    private final float h0 = 3.52f; //height of each layer in the building
    private final float temperature = 27.0f;
    private final float segma = 1/273.15f;
    private final float eqConst = 18410.183f;
    private boolean areFilesCreated;


    public FloorDetection()
    {
        areFilesCreated = false;
        avgReadings = new Vector<Float>();
        createFiles();
    }

    public int GetFloorNum (float pressure) {

        if(avgReadings.size() < timeToAverage)
        {
            avgReadings.add(pressure); //pt
        }
        else {
            avgT_0 = 0;
            for (float value :avgReadings) {
                avgT_0 += value;
            }
            avgT_0 /= (float) avgReadings.size(); //pt average
            if(firstRun)
            {
                avgT_2 = avgT_0;
                firstRun = false;
            }
            avgReadings.clear();
            if(currentTime == 2)
            {
                if (!enterLoop2 ) {
                    if (Math.abs(avgT_0 - avgT_2) > thetaT) {
                        if (num1 == 0)
                            avgT_5 = avgT_0;
                        if (num1 == N0) {
                            pstart = avgT_5;
                            enterLoop2 = true;
                            num1 = 0;
                        }
                        else
                            ++num1;
                    } else {
                        num1 = 0;
                    }
                }else {
                    if (Math.abs(avgT_0 - avgT_2) < thetaT){
                        if (num2 == 0)
                            avgT_5 = avgT_0;
                        if (num2 == N1) {
                            pend = avgT_5;
                            enterLoop2 = false;
                            num2 = 0;
                            currentFloor += Math.round((1/h0) * eqConst * (1+segma * temperature) * Math.log10(pstart/pend));
                        }
                        else
                            ++num2;
                    }else {
                        num2 = 0;
                    }
                }
                avgT_2 = avgT_0;
                currentTime = 0;
            }
            ++currentTime;
        }
        dataFileWriter.writeToFile("Floor_Detect",
                (float) pressure,
                avgT_0,
                avgT_2,
                pstart,
                pend,
                (float) currentFloor);
        return 0;
    }
    private void createFiles() {
        if (!areFilesCreated) {
            try {
                dataFileWriter = new DataFileWriter(FOLDER_NAME, DATA_FILE_NAMES, DATA_FILE_HEADINGS);
            } catch (IOException e) {
                Log.e("SensorActivity", e.toString());
            }
            areFilesCreated = true;
        }
    }
}
