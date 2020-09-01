package com.example.stepdetectionandstepestimation.OrientationFusedKalman;

public class LinearAccelerationFusion extends LinearAcceleration {

    public LinearAccelerationFusion(OrientationFused orientationFused) {
        super(orientationFused);
    }

    @Override
    public float[] getGravity() {
        return GravityUtil.getGravityFromOrientation(filter.getOutput());
    }
}
