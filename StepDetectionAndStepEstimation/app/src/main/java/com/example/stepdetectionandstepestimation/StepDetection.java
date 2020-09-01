package com.example.stepdetectionandstepestimation;
import android.hardware.SensorManager;
import android.os.Bundle;

import org.ejml.simple.SimpleMatrix;
public class StepDetection{
    private int stepCounter = 0;
    //Global Variable
    private double[] quaternionMatrix;

    public StepDetection(){
        //defining snesors
        quaternionMatrix = new double[4];
    }

    public double[] getQuaternion(float[] orientationMatix) {
        double q0 = 1 - Math.pow(orientationMatix[0], 2) - Math.pow(orientationMatix[1], 2) - Math.pow(orientationMatix[2], 2);
        if (q0 > 0)
            quaternionMatrix[0] = (float) Math.sqrt(q0);
        else
            quaternionMatrix[0] = 0;
        quaternionMatrix[1] = orientationMatix[0];
        quaternionMatrix[2] = orientationMatix[1];
        quaternionMatrix[3] = orientationMatix[2];
        return quaternionMatrix;
    }

    private static double[][] getRotationMatrix(double[] quaternionVector) {
        double[][] rotationMatrix = {{1 - 2 * (Math.pow(quaternionVector[2], 2) + Math.pow(quaternionVector[3], 2)),
                2 * (quaternionVector[1] * quaternionVector[2] - quaternionVector[0] * quaternionVector[3]),
                2 * (quaternionVector[1] * quaternionVector[3] + quaternionVector[0] * quaternionVector[2])},
                {2 * (quaternionVector[1] * quaternionVector[2] + quaternionVector[0] * quaternionVector[3]),
                        1 - 2 * (Math.pow(quaternionVector[1], 2) + Math.pow(quaternionVector[3], 2)),
                        2 * (quaternionVector[2] * quaternionVector[3] - quaternionVector[0] * quaternionVector[1])},
                {2 * (quaternionVector[1] * quaternionVector[3] - quaternionVector[0] * quaternionVector[2]),
                        2 * (quaternionVector[2] * quaternionVector[3] + quaternionVector[0] * quaternionVector[1]),
                        1 - 2 * (Math.pow(quaternionVector[1], 2) + Math.pow(quaternionVector[2], 2))}};
        return rotationMatrix;
    }

    public float[][] GetBodyAcc(float[] Acc, double[] quaternionVector, float[] gravity) {
        double[][] rotationMatrix = getRotationMatrix(quaternionVector);
        SimpleMatrix m_R_M = new SimpleMatrix(rotationMatrix);
        double[][] acc_M = ExtraFunctions.vectorToMatrix(Acc);
        SimpleMatrix m_acc_M = new SimpleMatrix(acc_M);
        SimpleMatrix m_linearAcc = m_R_M.mult(m_acc_M);
        float[][] linearAcc = ExtraFunctions.denseMatrixToArray(m_linearAcc.getMatrix());
        linearAcc[0][0] = (float) (linearAcc[0][0] - gravity[0]);
        linearAcc[1][0] = (float) (linearAcc[1][0] - gravity[1]);
        linearAcc[2][0] = (float) (linearAcc[2][0] - gravity[2]);
        return linearAcc;
    }

    public void detectSteps(float[] linearAcc, int stateCase, int stepEpoch)
    {
        float accMag = linearAcc[0] * linearAcc[0] + linearAcc[1] * linearAcc[1] + linearAcc[2] * linearAcc[2];
        accMag = (float) Math.sqrt(accMag);



    }


}

/*

double[][] R_rp = {{Math.cos(G_p), Math.sin(G_p) * Math.sin(G_r), Math.sin(G_p) * Math.cos(G_r)},
                            {0, Math.cos(G_r), -Math.sin(G_r)},
                            {-Math.sin(G_p), Math.cos(G_p) * Math.sin(G_r), Math.cos(G_p) * Math.cos(G_r)}}; //equation18

//        //remove bias from magnetic field initial values
//        double[][] M_init_unbiased = ExtraFunctions.vectorToMatrix(removeBias(M_m_values, M_m_bias));

        //convert arrays to matrices to allow for multiplication
        SimpleMatrix m_R_rp = new SimpleMatrix(R_rp);
 */
