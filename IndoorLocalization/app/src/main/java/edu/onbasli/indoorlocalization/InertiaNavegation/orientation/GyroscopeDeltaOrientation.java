package edu.onbasli.indoorlocalization.InertiaNavegation.orientation;

import edu.onbasli.indoorlocalization.InertiaNavegation.extra.ExtraFunctions;

import static android.util.Half.EPSILON;
import static java.lang.Math.cos;
import static java.lang.Math.sin;
import static java.lang.Math.sqrt;

public class GyroscopeDeltaOrientation {

    private boolean isFirstRun;
    private float sensitivity;
    private float lastTimestamp;
    private float[] gyroBias;
    private float[] calibPrevGyro;

    public GyroscopeDeltaOrientation() {
        this.gyroBias = new float[3];
        this.sensitivity = 0.0025f;
        this.isFirstRun = true;
        calibPrevGyro = new float[3];
    }

    public GyroscopeDeltaOrientation(float sensitivity, float[] gyroBias) {
        this();
        this.sensitivity = sensitivity;
        this.gyroBias = gyroBias;
        calibPrevGyro = new float[3];
    }

    public float[] calcDeltaOrientation(long timestamp, float[] rawGyroValues) {
        //get the first timestamp
        if (isFirstRun) {
            isFirstRun = false;
            lastTimestamp = ExtraFunctions.nsToSec(timestamp);
            return new float[3];
        }

        float[] unbiasedGyroValues = removeBias(rawGyroValues);
        //calibPrevGyro = unbiasedGyroValues;
        //float[] deltaOriantation = integrateAvgValues(timestamp, unbiasedGyroValues);
        // Return deltaOrientation[]
        return integrateValues(timestamp, unbiasedGyroValues);
    }

    public void setBias(float[] gyroBias){
        this.gyroBias = gyroBias;
    }

    private float[] removeBias(float[] rawGyroValues) {
        //ignoring the last 3 values of TYPE_UNCALIBRATED_GYROSCOPE, since the are only the Android-calculated biases
        float[] unbiasedGyroValues = new float[3];

        for (int i = 0; i < 3; i++)
            unbiasedGyroValues[i] = rawGyroValues[i] - gyroBias[i];

        //TODO: check how big of a difference this makes
        //applying a quick high pass filter
        for (int i = 0; i < 3; i++)
            if (Math.abs(unbiasedGyroValues[i]) > sensitivity)
                unbiasedGyroValues[i] = unbiasedGyroValues[i];
            else
                unbiasedGyroValues[i] = 0;

        return unbiasedGyroValues;
    }

    private float[] integrateValues(long timestamp, float[] gyroValues) {
        double currentTime = ExtraFunctions.nsToSec(timestamp);
        double deltaTime = currentTime - lastTimestamp;

        float[] deltaOrientation = new float[3];

        //integrating angular velocity with respect to time
        for (int i = 0; i < 3; i++)
            deltaOrientation[i] = gyroValues[i] * (float)deltaTime;

        lastTimestamp = (float) currentTime;

        return deltaOrientation;
    }
    /*private float[] deltaQuaternion(long timestamp, float[] gyroValues){
        double currentTime = ExtraFunctions.nsToSec(timestamp);
        double deltaTime = currentTime - lastTimestamp;
        // Calculate the angular speed of the sample
        float omegaMagnitude = (float) sqrt(gyroValues[0]*gyroValues[0] + gyroValues[1]*gyroValues[1] + gyroValues[2]*gyroValues[2]);
        // Normalize the rotation vector if it's big enough to get the axis
        // (that is, EPSILON should represent your maximum allowable margin of error)
        if (omegaMagnitude > EPSILON) {
            gyroValues[0] /= omegaMagnitude;
            gyroValues[1] /= omegaMagnitude;
            gyroValues[2] /= omegaMagnitude;
            double thetaOverTwo = omegaMagnitude * deltaTime / 2.0f;
            double sinThetaOverTwo = sin(thetaOverTwo);
            float cosThetaOverTwo = (float) cos(thetaOverTwo);
            deltaRotationVector[0] = sinThetaOverTwo * gyroValues[0];
            deltaRotationVector[1] = sinThetaOverTwo * gyroValues[1];
            deltaRotationVector[2] = sinThetaOverTwo * gyroValues[2];
            deltaRotationVector[3] = cosThetaOverTwo;
            float[] deltaRotationMatrix = new float[9];
            return SensorManager.getRotationMatrixFromVector(deltaRotationMatrix, deltaRotationVector);
        }
    }*/
    //Source: A Fusion Method for Combining Low-Cost
    //IMU/Magnetometer Outputs for Use in
    //Applications on Mobile Devices
   /* private float[] integrateAvgValues(long timestamp, float[] gyroValues) {
        double currentTime = ExtraFunctions.nsToSec(timestamp);
        double deltaTime = currentTime - lastTimestamp;

        float[] deltaOrientation = new float[3];
        //integrating angular velocity with respect to time
        for (int i = 0; i < 3; i++) {
            gyroValues[i] = (gyroValues[i] + calibPrevGyro[i]) / 2.0f;
            deltaOrientation[i] = gyroValues[i] * (float) deltaTime;
        }
        lastTimestamp = (float) currentTime;

        return deltaOrientation;
    }*/

}
