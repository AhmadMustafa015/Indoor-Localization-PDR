package com.example.stepdetectionandstepestimation.OrientationFusedKalman;

import org.apache.commons.math3.filter.ProcessModel;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class RotationProcessModel implements ProcessModel {
    private RealMatrix stateTransitionMatrix;

    /** The process noise covariance matrix. */
    private RealMatrix processNoiseCovMatrix;

    /** The initial state estimation of the observed process. */
    private RealVector initialStateEstimateVector;

    /** The initial error covariance matrix of the observed process. */
    private RealMatrix initialErrorCovMatrix;

    /** The initial error covariance matrix of the observed process. */
    private RealMatrix controlMatrix;

    public RotationProcessModel()
    {
        super();

        // A = stateTransitionMatrix
        stateTransitionMatrix = new Array2DRowRealMatrix(new double[][]
                {
                        { 1, 0, 0, 0 },
                        { 0, 1, 0, 0 },
                        { 0, 0, 1, 0 },
                        { 0, 0, 0, 1 } });

        // B = stateTransitionMatrix
        controlMatrix = new Array2DRowRealMatrix(new double[][]
                {
                        { 1, 0, 0, 0 },
                        { 0, 1, 0, 0 },
                        { 0, 0, 1, 0 },
                        { 0, 0, 0, 1 } });

        // Q = processNoiseCovMatrix
        processNoiseCovMatrix = new Array2DRowRealMatrix(new double[][]
                {
                        { 1, 0, 0, 0 },
                        { 0, 1, 0, 0 },
                        { 0, 0, 1, 0 },
                        { 0, 0, 0, 1 } });

        // xP = initialStateEstimateVector
        initialStateEstimateVector = new ArrayRealVector(new double[]
                { 0, 0, 0, 0 });

        // P0 = initialErrorCovMatrix;
        initialErrorCovMatrix = new Array2DRowRealMatrix(new double[][]
                {
                        { 0.1, 0, 0, 0 },
                        { 0, 0.1, 0, 0 },
                        { 0, 0, 0.1, 0 },
                        { 0, 0, 0, 0.1 } });
    }

    /** {@inheritDoc} */
    public RealMatrix getStateTransitionMatrix()
    {
        stateTransitionMatrix = new Array2DRowRealMatrix(new double[][]
                {
                        { 1, 0, 0, 0 },
                        { 0, 1, 0, 0 },
                        { 0, 0, 1, 0 },
                        { 0, 0, 0, 1 } });

        return stateTransitionMatrix;
    }

    /** {@inheritDoc} */
    public RealMatrix getControlMatrix()
    {
        return controlMatrix;
    }

    /** {@inheritDoc} */
    public RealMatrix getProcessNoise()
    {
        return processNoiseCovMatrix;
    }

    /** {@inheritDoc} */
    public RealVector getInitialStateEstimate()
    {
        return initialStateEstimateVector;
    }

    /** {@inheritDoc} */
    public RealMatrix getInitialErrorCovariance()
    {
        return initialErrorCovMatrix;
    }
}
