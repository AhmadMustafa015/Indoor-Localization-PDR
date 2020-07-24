package edu.onbasli.indoorlocalization.InertiaNavegation.extra;

import org.ejml.simple.SimpleMatrix;

public class KalmanFilter {
    private KalmanFilter(){}
    private float lastTimestamp;
    private static SimpleMatrix m_rotationNED = new SimpleMatrix(new double[][]{{0,1,0},
                                                                                {1,0,0},
                                                                                {0,0,-1}});
    private float[] xhatminus_accelerometerDCM_k; //a priori estimate of x
    private double[] xhat_accelerometerDCM_k;   //a posteri estimate of x
    private float[][] B_part1;
    private float[] uPrev_part1;
    private float[] wPrev_part1;
    private float[] horizontalOrientation_1; //R11 , R21
    private float[] horizontalOrientation_2; //R12, R22
    private float[] horizontalOrientation_3; //R13, R23
    private KalmanFilter(float[] data)
    {
        xhatminus_accelerometerDCM_k = new float[3];
        horizontalOrientation_1 = new float[2];
        horizontalOrientation_2 = new float[2];
        horizontalOrientation_3 = new float[2];
        xhat_accelerometerDCM_k = new double[3];
    }
    private void initiateMatrix(float[][] DCM, long timeStamp){
        //for part1
        xhatminus_accelerometerDCM_k[0] = DCM[2][0]; //R31
        xhatminus_accelerometerDCM_k[0] = DCM[2][1]; //R32
        xhatminus_accelerometerDCM_k[0] = DCM[2][2]; //R33
        //for part2
        horizontalOrientation_1[0] = DCM[0][0]; //R11
        horizontalOrientation_1[1] = DCM[1][0]; //R21
        horizontalOrientation_2[0] = DCM[0][1]; //R12
        horizontalOrientation_2[0] = DCM[1][1]; //R22
        horizontalOrientation_3[0] = DCM[0][2]; //R13
        horizontalOrientation_3[1] = DCM[1][2]; //R23
    }
    private void initiateExogenousControl(float[] biasedRawGyro)
    {
        uPrev_part1[0] = biasedRawGyro[0];
        uPrev_part1[1] = biasedRawGyro[1];
        uPrev_part1[2] = biasedRawGyro[2];
    }
    private void timeUpdate()
    {
        double[][] xhat_d = ExtraFunctions.vectorToMatrix(xhat_accelerometerDCM_k);
        SimpleMatrix xhat = new SimpleMatrix(xhat_d);
    }
}
/*

n_iter = 50
sz = (n_iter,) # size of array
x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
z = np.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)

Q = 1e-5 # process variance

# allocate space for arrays
xhat=np.zeros(sz)      # a posteri estimate of x
P=np.zeros(sz)         # a posteri error estimate
xhatminus=np.zeros(sz) # a priori estimate of x
Pminus=np.zeros(sz)    # a priori error estimate
K=np.zeros(sz)         # gain or blending factor

R = 0.1**2 # estimate of measurement variance, change to see effect

# intial guesses
xhat[0] = 0.0
P[0] = 1.0
 */