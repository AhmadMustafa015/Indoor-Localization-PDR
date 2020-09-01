package com.example.stepdetectionandstepestimation.OrientationFusedKalman;

public abstract class LinearAcceleration {

    private static final String tag = LinearAcceleration.class.getSimpleName();

    private final float[] output = new float[]{0, 0, 0};

    protected final BaseFilter filter;

    public LinearAcceleration(BaseFilter filter) {
        this.filter = filter;
    }

    public float[] filter(float[] values) {

        float[] gravity = getGravity();

        // Determine the linear acceleration
        output[0] = values[0] - gravity[0];
        output[1] = values[1] - gravity[1];
        output[2] = values[2] - gravity[2];

        return output;
    }

    public abstract float[] getGravity();
}
