package edu.onbasli.indoorlocalization.InertiaNavegation.orientation;

import org.ejml.simple.SimpleMatrix;

import edu.onbasli.indoorlocalization.InertiaNavegation.extra.ExtraFunctions;

public final class MagneticFieldOrientation {

    private MagneticFieldOrientation() {}
    //This application note uses the industry standard “NED” (North, East, Down) coordinate system to label
    //axes on the mobile phone and the IMU use ENU a simple conversion by using this matrix
    private static SimpleMatrix m_rotationNED = new SimpleMatrix(new double[][]{{0,1,0},
                                                                                {1,0,0},
                                                                                {0,0,-1}});

    public static float[][] getOrientationMatrix(float[] G_values, float[] M_values, float[] M_bias) {

        //G = Gyroscope, M = Magnetic Field
        //m = matrix
        //where r = roll, p = pitch, h = heading (yaw)

        //remove bias from magnetic field initial values
        double[][] M_init_unbiased = ExtraFunctions.vectorToMatrix(removeBias(M_values, M_bias));

        SimpleMatrix m_M_init_unbiased = new SimpleMatrix(M_init_unbiased);
        SimpleMatrix m_G_values = new SimpleMatrix(ExtraFunctions.vectorToMatrix(ExtraFunctions.floatVectorToDoubleVector(G_values)));

        m_M_init_unbiased =  m_rotationNED.mult(m_M_init_unbiased);
        m_G_values =  m_rotationNED.mult(m_G_values);

        float[][] G_m_values = ExtraFunctions.denseMatrixToArray(m_G_values.getMatrix());

        //calculate roll and pitch from gravity
        double G_r = Math.atan2(G_m_values[1][0], G_m_values[2][0]);
        double G_p = Math.atan2(-G_m_values[0][0], G_m_values[1][0] * Math.sin(G_r) + G_m_values[2][0] * Math.cos(G_r));

        //create the rotation matrix representing the roll and pitch
        double[][] R_rp = {{Math.cos(G_p), Math.sin(G_p) * Math.sin(G_r), Math.sin(G_p) * Math.cos(G_r)},
                            {0, Math.cos(G_r), -Math.sin(G_r)},
                            {-Math.sin(G_p), Math.cos(G_p) * Math.sin(G_r), Math.cos(G_p) * Math.cos(G_r)}}; //equation18

        //convert arrays to matrices to allow for multiplication
        SimpleMatrix m_R_rp = new SimpleMatrix(R_rp);

        //rotate magnetic field values in accordance to gravity readings
        SimpleMatrix m_M_rp = m_R_rp.mult(m_M_init_unbiased);

        //calc heading (rads) from rotated magnetic field and added magnetic declination angle
        double h = -1*(Math.atan2(-m_M_rp.get(1), m_M_rp.get(0)) + 11.0 * Math.PI/180.0);

        //rotation matrix representing heading, is negative when moving East of North
        double[][] R_h = {{Math.cos(h), -Math.sin(h), 0},
                          {Math.sin(h),  Math.cos(h), 0},
                          {0,            0,           1}};

        //calc complete (initial) rotation matrix by multiplying roll/pitch matrix with heading matrix
        SimpleMatrix m_R_h = new SimpleMatrix(R_h);
        SimpleMatrix m_R = m_R_rp.mult(m_R_h); //DCM DCM = R11 R12 R13
                                                        // R21 R22 R23
                                                        // R31 R32 R33


        return ExtraFunctions.denseMatrixToArray(m_R.getMatrix());

    }

    public static float getHeading(float[][] orientationMatrix) {
        //float[][] orientationMatrix = getOrientationMatrix(G_values, M_values, M_bias);
        return (float) Math.atan2(orientationMatrix[1][0], orientationMatrix[0][0]);
    }

//    private static double[] removeBias(float[][] M_init, float[][] M_bias) {
//        //ignoring the last 3 values of M_init, which are the android-calculated iron biases
//        double[] M_biasRemoved = new double[3];
//        for (int i = 0; i < 3; i++)
//            M_biasRemoved[i] = M_init[i][0] - M_bias[i][0];
//        return M_biasRemoved;
//    }

    private static double[] removeBias(float[] M_init, float[] M_bias) {
        //ignoring the last 3 values of M_init, which are the android-calculated iron biases
        double[] M_biasRemoved = new double[3];
        for (int i = 0; i < 3; i++)
            M_biasRemoved[i] = M_init[i] - M_bias[i];
        return M_biasRemoved;
    }

}
