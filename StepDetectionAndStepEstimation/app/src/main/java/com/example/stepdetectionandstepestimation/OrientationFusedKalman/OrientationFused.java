package com.example.stepdetectionandstepestimation.OrientationFusedKalman;
import org.apache.commons.math3.complex.Quaternion;

public abstract class OrientationFused extends BaseFilter {
    protected static final float EPSILON = 0.000000001f;
    // Nano-second to second conversion
    protected static final float NS2S = 1.0f / 1000000000.0f;
    private static final String tag = OrientationFused.class.getSimpleName();
    public static float DEFAULT_TIME_CONSTANT = 0.18f;
    // The coefficient for the fusedOrientation... 0.5 = means it is averaging the two
    // transfer functions (rotations from the gyroscope and
    // acceleration/magnetic, respectively).
    public float timeConstant;
    protected Quaternion rotationVector;
    protected long timestamp = 0;
    protected float[] output = new float[3];

    /**
     * Initialize a singleton instance.
     */
    public OrientationFused() {
        this(DEFAULT_TIME_CONSTANT);
    }

    public OrientationFused(float timeConstant) {
        this.timeConstant = timeConstant;

    }

    @Override
    public float[] getOutput() {
        return output;
    }

    public void reset() {
        timestamp = 0;
        rotationVector = null;
    }

    public boolean isBaseOrientationSet() {
        return rotationVector != null;
    }

    /**
     * The complementary fusedOrientation coefficient, a floating point value between 0-1,
     * exclusive of 0, inclusive of 1.
     *
     * @param timeConstant
     */
    public void setTimeConstant(float timeConstant) {
        this.timeConstant = timeConstant;
    }

    /**
     * Calculate the fused orientation of the device.
     * @param gyroscope the gyroscope measurements.
     * @param timestamp the gyroscope timestamp
     * @param acceleration the acceleration measurements
     * @param magnetic the magnetic measurements
     * @return the fused orientation estimation.
     */
    public abstract float[] calculateFusedOrientation(float[] gyroscope, long timestamp, float[] acceleration, float[] magnetic);

    public void setBaseOrientation(Quaternion baseOrientation) {
        rotationVector = baseOrientation;
    }
}
