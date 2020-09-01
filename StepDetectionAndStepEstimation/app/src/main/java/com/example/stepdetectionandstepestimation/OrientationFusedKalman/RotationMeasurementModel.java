package com.example.stepdetectionandstepestimation.OrientationFusedKalman;

import org.apache.commons.math3.filter.MeasurementModel;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class RotationMeasurementModel implements MeasurementModel
{
    private final double noiseCoefficient  = 0.001;

    /**
     * The measurement matrix, used to associate the measurement vector to the
     * internal state estimation vector.
     */
    private RealMatrix measurementMatrix;

    /**
     * The measurement noise covariance matrix.
     */
    private RealMatrix measurementNoise;

    public RotationMeasurementModel()
    {
        super();

        // H = measurementMatrix
        measurementMatrix = new Array2DRowRealMatrix(new double[][]
                {
                        { 1, 0, 0, 0 },
                        { 0, 1, 0, 0 },
                        { 0, 0, 1, 0 },
                        { 0, 0, 0, 1 } });

        // R = measurementNoise
        measurementNoise = new Array2DRowRealMatrix(new double[][]
                {
                        { noiseCoefficient, 0, 0, 0 },
                        { 0, noiseCoefficient, 0, 0 },
                        { 0, 0, noiseCoefficient, 0 },
                        { 0, 0, 0, noiseCoefficient } });
    }

    /** {@inheritDoc} */
    public RealMatrix getMeasurementMatrix()
    {
        return measurementMatrix;
    }

    /** {@inheritDoc} */
    public RealMatrix getMeasurementNoise()
    {
        return measurementNoise;
    }
}