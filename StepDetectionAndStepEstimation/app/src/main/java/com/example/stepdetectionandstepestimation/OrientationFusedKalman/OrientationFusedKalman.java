package com.example.stepdetectionandstepestimation.OrientationFusedKalman;
import android.util.Log;

import org.apache.commons.math3.complex.Quaternion;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;

public class OrientationFusedKalman extends OrientationFused {
    private static final String TAG = OrientationFusedComplementary.class.getSimpleName();

    private final RotationKalmanFilter kalmanFilter;
    private final AtomicBoolean run;
    private volatile float dT;
    private volatile double[] quaternion = new double[4];
    private volatile float[] output = new float[3];
    private Thread thread;

    private volatile Quaternion rotationVectorAccelerationMagnetic;
    private final double[] vectorGyroscope = new double[4];
    private final double[] vectorAccelerationMagnetic = new double[4];

    public OrientationFusedKalman() {
        this(DEFAULT_TIME_CONSTANT);
    }

    public OrientationFusedKalman(float timeConstant) {
        super(timeConstant);
        run = new AtomicBoolean(false);
        kalmanFilter = new RotationKalmanFilter(new RotationProcessModel(), new RotationMeasurementModel());
    }

    public void startFusion() {
        if (!run.get() && thread == null) {
            run.set(true);

            thread = new Thread(new Runnable() {
                @Override
                public void run() {
                    while (run.get() && !thread.interrupted()) {

                        output = calculate();

                        try {
                            thread.sleep(20);
                        } catch (InterruptedException e) {
                            Log.e(TAG, "Kalman Thread", e);
                            thread.currentThread().interrupt();
                        }
                    }

                    thread.currentThread().interrupt();
                }
            });

            thread.start();
        }
    }

    public void stopFusion() {
        if (run.get() && thread != null) {
            run.set(false);
            thread.interrupt();
            thread = null;
        }
    }

    public float[] getOutput() {
        return output;
    }
    public double[] getQuaternion() { return quaternion;}
    private void setQuaternion(Quaternion filteredQuaternion) {
        quaternion[0] = filteredQuaternion.getQ0();
        quaternion[1] = filteredQuaternion.getQ1();
        quaternion[2] = filteredQuaternion.getQ2();
        quaternion[3] = filteredQuaternion.getQ3();
    }

    /**
     * Calculate the fused orientation of the device.
     *
     * Rotation is positive in the counterclockwise direction (right-hand rule). That is, an observer looking from some positive location on the x, y, or z axis at
     * a device positioned on the origin would report positive rotation if the device appeared to be rotating counter clockwise. Note that this is the
     * standard mathematical definition of positive rotation and does not agree with the aerospace definition of roll.
     *
     * See: https://source.android.com/devices/sensors/sensor-types#rotation_vector
     *
     * Returns a vector of size 3 ordered as:
     * [0]X points east and is tangential to the ground.
     * [1]Y points north and is tangential to the ground.
     * [2]Z points towards the sky and is perpendicular to the ground.
     *
     * @return An orientation vector -> @link SensorManager#getOrientation(float[], float[])}
     */
    private float[] calculate() {
        if (rotationVector != null && rotationVectorAccelerationMagnetic != null && dT != 0) {
            vectorGyroscope[0] = (float) rotationVector.getVectorPart()[0];
            vectorGyroscope[1] = (float) rotationVector.getVectorPart()[1];
            vectorGyroscope[2] = (float) rotationVector.getVectorPart()[2];
            vectorGyroscope[3] = (float) rotationVector.getScalarPart();

            vectorAccelerationMagnetic[0] = (float) rotationVectorAccelerationMagnetic.getVectorPart()[0];
            vectorAccelerationMagnetic[1] = (float) rotationVectorAccelerationMagnetic.getVectorPart()[1];
            vectorAccelerationMagnetic[2] = (float) rotationVectorAccelerationMagnetic.getVectorPart()[2];
            vectorAccelerationMagnetic[3] = (float) rotationVectorAccelerationMagnetic.getScalarPart();

            // Apply the Kalman fusedOrientation... Note that the prediction and correction
            // inputs could be swapped, but the fusedOrientation is much more stable in this
            // configuration.
            kalmanFilter.predict(vectorGyroscope);
            kalmanFilter.correct(vectorAccelerationMagnetic);

            // rotation estimation.
            Quaternion result = new Quaternion(kalmanFilter.getStateEstimation()[3], Arrays.copyOfRange(kalmanFilter.getStateEstimation(), 0, 3));
            setQuaternion(result);
            output = AngleUtils.getAngles(result.getQ0(), result.getQ1(), result.getQ2(), result.getQ3());
        }

        return output;
    }

    /**
     * Calculate the fused orientation of the device.
     *
     * @param gyroscope    the gyroscope measurements.
     * @param timestamp    the gyroscope timestamp
     * @param acceleration the acceleration measurements
     * @param magnetic     the magnetic measurements
     * @return the fused orientation estimation.
     */
    public float[] calculateFusedOrientation(float[] gyroscope, long timestamp, float[] acceleration, float[] magnetic) {
        if(isBaseOrientationSet()) {
            if (this.timestamp != 0) {
                dT = (timestamp - this.timestamp) * NS2S;

                rotationVectorAccelerationMagnetic = RotationUtil.getOrientationVectorFromAccelerationMagnetic(acceleration, magnetic);
                rotationVector = RotationUtil.integrateGyroscopeRotation(rotationVector, gyroscope, dT, EPSILON);
            }
            this.timestamp = timestamp;

            return output;
        }  else {
            throw new IllegalStateException("You must call setBaseOrientation() before calling calculateFusedOrientation()!");
        }
    }
}
