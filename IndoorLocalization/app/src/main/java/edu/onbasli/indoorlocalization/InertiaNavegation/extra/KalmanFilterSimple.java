package edu.onbasli.indoorlocalization.InertiaNavegation.extra;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.simple.SimpleMatrix;
public class KalmanFilterSimple {

    // kinematics description
    private SimpleMatrix A,Q,H,B;
    private double q = 1e-5;
    private int matrix_size;
    private boolean firstRun;
    // system state estimate
    private SimpleMatrix x,P,u,w;
    public void configure(int size_matrix) {
        if(size_matrix ==3) {
            this.A = new SimpleMatrix(new double[][]{{1, 0, 0},
                    {0, 1, 0},
                    {0, 0, 1}});
            this.H = new SimpleMatrix(new double[][]{{1, 0, 0},
                    {0, 1, 0},
                    {0, 0, 1}});
        } else
        {
            this.A = new SimpleMatrix(new double[][]{{1, 0},
                    {0, 1}});
            this.H = new SimpleMatrix(new double[][]{{1, 0},
                    {0, 1}});
        }
        this.matrix_size = size_matrix;
        //this.Q = new SimpleMatrix(ExtraFunctions.vectorToMatrix(new double[]{q,q,q}));
        firstRun = true;
        //this.H = new SimpleMatrix(H);
    }
    // initiate x to [R31, R32, R33]
    public void setState(double[][] x) {
        this.x = new SimpleMatrix(x);
        //this.P = new SimpleMatrix(P);
    }
    public void initiateErrorCovariance(double[] newR) {
        float[][] x_m = ExtraFunctions.denseMatrixToArray((x.getMatrix()));
        double[][] p;
        if (matrix_size == 3) {
            p = new double[][]{{(newR[0] - x_m[0][0]) * (newR[0] - x_m[0][0]), (newR[0] - x_m[0][0]) * (newR[1] - x_m[1][0]), (newR[0] - x_m[0][0]) * (newR[2] - x_m[2][0])},
                    {(newR[1] - x_m[1][0]) * (newR[0] - x_m[0][0]), (newR[1] - x_m[1][0]) * (newR[1] - x_m[1][0]), (newR[2] - x_m[2][0]) * (newR[1] - x_m[1][0])},
                    {(newR[0] - x_m[0][0]) * (newR[2] - x_m[2][0]), (newR[1] - x_m[1][0]) * (newR[2] - x_m[2][0]), (newR[2] - x_m[2][0]) * (newR[2] - x_m[2][0])}};
        }else {
            p = new double[][]{{(newR[0] - x_m[0][0]) * (newR[0] - x_m[0][0]), (newR[0] - x_m[0][0]) * (newR[1] - x_m[1][0])},
                                {(newR[1] - x_m[1][0]) * (newR[0] - x_m[0][0]), (newR[1] - x_m[1][0]) * (newR[1] - x_m[1][0])}};
        }
        this.P = new SimpleMatrix(p);
    }
    public void setControlMatrix(double[][] U,double[][] B) {

        this.B = new SimpleMatrix(B);
        this.u = new SimpleMatrix(U);
    }

    public void predict() {
        // x = A x
        x = A.mult(x).plus(B.mult(u));

        // P = A P A' + Q
        P = A.mult(P).mult(A.transpose()).plus(q);
    }

    public void update(double[] Ri, double[] Z,double[][] _R) {
        // a fast way to make the matrices usable by SimpleMatrix
        SimpleMatrix Ri_m = new SimpleMatrix(ExtraFunctions.vectorToMatrix(Ri));
        SimpleMatrix Z_m = new SimpleMatrix(ExtraFunctions.vectorToMatrix(Z));
        SimpleMatrix z = H.mult(Ri_m).plus(Z_m);
        SimpleMatrix R = new SimpleMatrix(_R);

        // y = z - H x
        SimpleMatrix y = z.minus(H.mult(x));
        // S = H P H' + R
        SimpleMatrix S = H.mult(P).mult(H.transpose()).plus(R);
        // K = PH'S^(-1)
        SimpleMatrix K = P.mult(H.transpose().mult(S.invert()));

        // x = x + Ky
        x = x.plus(K.mult(y));

        // P = (I-kH)P = P - KHP
        P = P.minus(K.mult(H).mult(P));
    }

    public float[][] getState() {
        return ExtraFunctions.denseMatrixToArray(x.getMatrix());
    }

    public float[][] getCovariance() {
        return ExtraFunctions.denseMatrixToArray(P.getMatrix());
    }
}
