package com.example.stepdetectionandstepestimation.OrientationFusedKalman;

import android.hardware.SensorManager;

public class GravityUtil {

    private static final String tag = GravityUtil.class.getSimpleName();

    private static float[] gravity = new float[]{SensorManager.GRAVITY_EARTH, SensorManager.GRAVITY_EARTH, SensorManager.GRAVITY_EARTH};

    /**
     * Assumes a positive, counter-clockwise, right-handed rotation
     * orientation[0] = pitch, rotation around the X axis.
     * orientation[1] = roll, rotation around the Y axis
     * orientation[2] = azimuth, rotation around the Z axis
     * @param orientation The orientation.
     * @return The gravity components of the orientation.
     */
    public static float[] getGravityFromOrientation(float[] orientation) {

        float[] components = new float[3];

        float pitch = orientation[1];
        float roll = orientation[2];

        // Find the gravity component of the X-axis
        // = g*-cos(pitch)*sin(roll);
        components[0] = (float) -(gravity[0]
                * -Math.cos(pitch) * Math
                .sin(roll));

        // Find the gravity component of the Y-axis
        // = g*-sin(pitch);
        components[1] = (float) (gravity[1] * -Math
                .sin(pitch));

        // Find the gravity component of the Z-axis
        // = g*cos(pitch)*cos(roll);
        components[2] = (float) (gravity[2]
                * Math.cos(pitch) * Math.cos(roll));

        return components;
    }

    /**
     * Set the gravity as measured by the sensor. Defaults to SensorManager.GRAVITY_EARTH
     * @param g the gravity of earth in units of m/s^2
     */
    public static void setGravity(float[] g) {
        gravity = g;
    }
}
