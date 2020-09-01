package com.example.stepdetectionandstepestimation.filters;


import com.example.stepdetectionandstepestimation.OrientationFusedKalman.BaseFilter;

public abstract class AveragingFilter extends BaseFilter {
    public static float DEFAULT_TIME_CONSTANT = 0.18f;

    protected float timeConstant;
    protected long startTime;
    protected long timestamp;
    protected int count;

    public AveragingFilter() {
        this(DEFAULT_TIME_CONSTANT);
    }

    public AveragingFilter(float timeConstant) {
        this.timeConstant = timeConstant;
        reset();
    }

    public void reset() {
        startTime = 0;
        timestamp = 0;
        count = 0;
    }

    public abstract float[] filter(float[] data);
}
