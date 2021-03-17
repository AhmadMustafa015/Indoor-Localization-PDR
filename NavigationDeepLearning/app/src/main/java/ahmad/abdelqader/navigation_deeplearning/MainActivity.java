package ahmad.abdelqader.navigation_deeplearning;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.renderscript.Matrix3f;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TableRow;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.res.ResourcesCompat;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Queue;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;
import org.apache.commons.math3.complex.Quaternion;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import com.google.android.material.floatingactionbutton.FloatingActionButton;

import ahmad.abdelqader.navigation_deeplearning.graph.ScatterPlot;


public class MainActivity extends AppCompatActivity implements SensorEventListener, TextToSpeech.OnInitListener {


    private static final int N_SAMPLES = 162;
    private static int prevIdx = -1;
    private float totalDistance = 0;

    private static List<Float> ax;
    private static List<Float> ay;
    private static List<Float> az;

    private static List<Float> mx;
    private static List<Float> my;
    private static List<Float> mz;

    private static List<Float> gx;
    private static List<Float> gy;
    private static List<Float> gz;

    private static List<Float> ma;
    private static List<Float> ml;
    private static List<Float> mg;
    private SensorManager mSensorManager;
    private Sensor mAccelerometer;
    private Sensor mGyroscope;
    private Sensor mMagnetic;
    private int stride = 10;
    private int stride_gyro = 0;
    private int stride_acc = 0;

    private boolean first_run = true;
    private TextView downstairsTextView;
    private TextView runningTextView;
    private TextView standingTextView;
    private TextView upstairsTextView;
    private TextView walkingTextView;
    private TextView elevatorTextView;
    private TextView fallingTextView;
    private TextView carTextView;
    private TextView accX;
    private TextView accY;
    private TextView accZ;
    private TextView linAccX;
    private TextView linAccY;
    private TextView linAccZ;
    private TextView gyroX;
    private TextView gyroY;
    private TextView gyroZ;
    private TextView totalD;

    private TableRow downstairsTableRow;
    private TableRow runningTableRow;
    private TableRow standingTableRow;
    private TableRow upstairsTableRow;
    private TableRow walkingTableRow;
    private TableRow elevatorTableRow;
    private TableRow fallingTableRow;
    private TableRow carTableRow;
    private TextToSpeech textToSpeech;


    private Quaternion init_q;
    private float[] init_p;
    private float[] current_p;
    private Quaternion current_q;
    private ScatterPlot scatterPlot;
    private LinearLayout mLinearLayout;
    float[][][] inputData_acc;
    float[][][] inputData_gyro;

    private float[] results;
    private HARClassifier classifier;
    private static final float Frequency = 50f;
    private static final float nyquestRate = Frequency / 2;
    private static final float cutoff_low = 0.3f;
    private static final float cutoff_high = 20f;
    private String[] labels = {"Walking", "Upstairs", "Downstairs", "Standing_Sitting", "Running", "Elevator", "Falling", "Car"};
    private boolean isRunning;
    private FloatingActionButton fabButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_graph);
        ax = new ArrayList<>();
        ay = new ArrayList<>();
        az = new ArrayList<>();
        mx = new ArrayList<>();
        my = new ArrayList<>();
        mz = new ArrayList<>();
        gx = new ArrayList<>();
        gy = new ArrayList<>();
        gz = new ArrayList<>();
        ma = new ArrayList<>();
        ml = new ArrayList<>();
        mg = new ArrayList<>();

        fabButton = findViewById(R.id.fab);
        mLinearLayout = findViewById(R.id.linearLayoutGraph);
        totalD = (TextView) findViewById(R.id.TV_totalDistace);
        inputData_acc  = new float[1][N_SAMPLES][3];
        inputData_gyro = new float[1][N_SAMPLES][3];


        init_p = new float[]{0, 0, 0};
        current_p = new float[3];


        scatterPlot = new ScatterPlot("Gyro");
        scatterPlot.addPoint(0, 0); //AM: future update sit value using GPS
        //scatterPlot.addPointM(0, 0); //AM: future update sit value using GPS
        //scatterPlot.addPointC(0, 0); //AM: future update sit value using GPS
        mLinearLayout.addView(scatterPlot.getGraphView(getApplicationContext()));


        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);

        mMagnetic = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);

        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        classifier = new HARClassifier(getApplicationContext());

        textToSpeech = new TextToSpeech(this, this);
        textToSpeech.setLanguage(Locale.US);

        fabButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

            if (!isRunning) {

                fabButton.setImageDrawable(ContextCompat.getDrawable(MainActivity.this, R.drawable.ic_pause_black_24dp));
                isRunning = true;
                mSensorManager.registerListener(MainActivity.this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
                mSensorManager.registerListener(MainActivity.this, mMagnetic, SensorManager.SENSOR_DELAY_FASTEST);
                mSensorManager.registerListener(MainActivity.this, mGyroscope, SensorManager.SENSOR_DELAY_FASTEST);


            } else {

                isRunning = false;
                mSensorManager.unregisterListener(MainActivity.this, mAccelerometer);
                mSensorManager.unregisterListener(MainActivity.this, mMagnetic);
                mSensorManager.unregisterListener(MainActivity.this, mGyroscope);
                fabButton.setImageDrawable(ContextCompat.getDrawable(MainActivity.this, R.drawable.ic_play_arrow_black_24dp));
            }

            }
        });
    }

    @Override
    public void onInit(int status) {
        Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                if (results == null || results.length == 0) {
                    return;
                }
                float max = -1;
                int idx = -1;
                for (int i = 0; i < results.length; i++) {
                    if (results[i] > max) {
                        idx = i;
                        max = results[i];
                    }
                }

                if (max > 0.50 && idx != prevIdx) {
                    textToSpeech.speak(labels[idx], TextToSpeech.QUEUE_ADD, null,
                            Integer.toString(new Random().nextInt()));
                    prevIdx = idx;
                }
            }
        }, 1000, 3000);
    }

    protected void onResume() {
        super.onResume();
        /*getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_FASTEST);
        //getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION), SensorManager.SENSOR_DELAY_FASTEST );
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_GYROSCOPE), SensorManager.SENSOR_DELAY_FASTEST);
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD), SensorManager.SENSOR_DELAY_FASTEST);
*/
    }
    protected void onStop() {
        super.onStop();
        mSensorManager.unregisterListener(MainActivity.this, mAccelerometer);
        mSensorManager.unregisterListener(MainActivity.this, mMagnetic);
        mSensorManager.unregisterListener(MainActivity.this, mGyroscope);
    }

    @Override
    public void onDestroy() {
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
        super.onDestroy();
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        activityPrediction();

        Sensor sensor = event.sensor;
        if (sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            ax.add(event.values[0]);
            ay.add(event.values[1]);
            az.add(event.values[2]);
            if(!first_run)
                stride_acc++;

        } else if (sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            gx.add(10 * event.values[0]);
            gy.add(10 * event.values[1]);
            gz.add(10 * event.values[2]);
            if(!first_run)
                stride_gyro++;

        } else if (sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD) {
            mx.add(event.values[0]);
            my.add(event.values[1]);
            mz.add(event.values[2]);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    private void activityPrediction() {

        if ((first_run && (ax.size() >= N_SAMPLES) && (gx.size() >= N_SAMPLES)) || (!first_run && stride_acc >= 10 && stride_gyro >= 10)) {
            if (first_run) {
                for (int i = 0; i < N_SAMPLES; i++) {
                    inputData_acc[0][i][0] = ax.get(i);
                    inputData_acc[0][i][1] = ay.get(i);
                    inputData_acc[0][i][2] = az.get(i);

                    inputData_gyro[0][i][0] = gx.get(i);
                    inputData_gyro[0][i][1] = gy.get(i);
                    inputData_gyro[0][i][2] = gz.get(i);


                }
                init_q = getOrientationVectorFromAccelerationMagnetic(new float[]{ax.get(ax.size() - 1), ay.get(ay.size() - 1), az.get(az.size() - 1)},
                        new float[]{mx.get(mx.size() - 1), my.get(my.size() - 1), mz.get(mz.size() - 1)});
                first_run = false;
                current_q = init_q;
                current_p[0] = 0;
                current_p[1] = 0;
                current_p[2] = 0;
            }
            else {
                for (int i = 0; i < stride; i++) {
                    inputData_acc[0][N_SAMPLES-1-stride+i][0] = ax.get(i);
                    inputData_acc[0][N_SAMPLES-1-stride+i][1] = ay.get(i);
                    inputData_acc[0][N_SAMPLES-1-stride+i][2] = az.get(i);

                    inputData_gyro[0][N_SAMPLES-1-stride+i][0] = gx.get(i);
                    inputData_gyro[0][N_SAMPLES-1-stride+i][1] = gy.get(i);
                    inputData_gyro[0][N_SAMPLES-1-stride+i][2] = gz.get(i);
                }
            }
            Object[] inputs = {inputData_acc, inputData_gyro};
            classifier.predictProbabilities(inputs);
            float[][] parsedOutput0 = classifier.getDisplacment();
            float[][] parsedOutput1 = classifier.getQuaternion();
            RealVector y_data_p = new ArrayRealVector(new double[]{parsedOutput0[0][0],parsedOutput0[0][1],parsedOutput0[0][2]}, false);
            RealMatrix rotation_matrix =  new Array2DRowRealMatrix(getRotationMatrix(new double[]{current_q.getQ0(),current_q.getQ1(),current_q.getQ2(),current_q.getQ3()}), false);
            //{parsedOutput1[0][0],parsedOutput1[0][1],parsedOutput1[0][2],parsedOutput1[0][3]}
            RealVector rotated_pos = rotation_matrix.operate(y_data_p);
            double [] pos_data = rotated_pos.toArray();
            totalDistance = (float) (totalDistance + Math.sqrt(pos_data[0] * pos_data[0] + pos_data[2] * pos_data[2] + pos_data[1] * pos_data[1]));
            current_p[0] = current_p[0] + (float) pos_data[0];
            current_p[1] = current_p[1] + (float) pos_data[1];
            current_p[2] = current_p[2] + (float) pos_data[2];
            Quaternion new_q = new Quaternion(parsedOutput1[0][0],parsedOutput1[0][1],parsedOutput1[0][2],parsedOutput1[0][3]);
            new_q = new_q.normalize();
            current_q = current_q.multiply(new_q);
            /*float max = -1;
            int idx = -1;
            for (int i = 0; i < results.length; i++)
            {
                if (results[i] > max) {
                    idx = i;
                    max = results[i];
                }
            }

            setProbabilities();
            setRowsColor(idx);*/

            //Plot

            float oPointX = scatterPlot.getLastYPoint();
            float oPointY = -scatterPlot.getLastXPoint();
                    /*float oPointXM = scatterPlot.getLastYPointM();
                    float oPointYM = -scatterPlot.getLastXPointM();
                    float oPointXC = scatterPlot.getLastYPointC();
                    float oPointYC = -scatterPlot.getLastXPointC();*/
            //calculating XY points from heading and stride_length
            //gyroHeading = (float) (Math.toDegrees(gyroHeading) + 360) % 360;
            //gyroHeading = (float) (Math.toRadians(gyroHeading));

            //rotating points by 90 degrees, so north is up
            float rPointX = current_p[0];
            float rPointY = current_p[1];

            scatterPlot.addPoint(rPointX, rPointY);
            mLinearLayout.removeAllViews();
            mLinearLayout.addView(scatterPlot.getGraphView(getApplicationContext()));
            totalD.setText(String.format("%.2f",totalDistance));

            ax.clear();
            ay.clear();
            az.clear();
            gx.clear();
            gy.clear();
            gz.clear();
            ma.clear();
            ml.clear();
            mg.clear();
            stride_gyro =0;
            stride_acc=0;
            for (int i = 0; i < (N_SAMPLES - stride); i++) {
                inputData_acc[0][i][0] = inputData_acc[0][stride+i][0];
                inputData_acc[0][i][1] = inputData_acc[0][stride+i][1];
                inputData_acc[0][i][2] = inputData_acc[0][stride+i][2];

                inputData_gyro[0][i][0] = inputData_gyro[0][stride+i][0];
                inputData_gyro[0][i][1] = inputData_gyro[0][stride+i][1];
                inputData_gyro[0][i][2] = inputData_gyro[0][stride+i][2];
            }

        }
    }

    private void setProbabilities() {
        walkingTextView.setText(Float.toString(round(results[0], 2)));
        upstairsTextView.setText(Float.toString(round(results[1], 2)));
        downstairsTextView.setText(Float.toString(round(results[2], 2)));
        standingTextView.setText(Float.toString(round(results[3], 2)));
        runningTextView.setText(Float.toString(round(results[4], 2)));
        elevatorTextView.setText(Float.toString(round(results[5], 2)));
        fallingTextView.setText(Float.toString(round(results[6], 2)));
        carTextView.setText(Float.toString(round(results[7], 2)));

    }

    /*private void setRowsColor(int idx) {
        downstairsTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        runningTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        standingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        upstairsTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        walkingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        elevatorTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        fallingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        carTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));

        if (idx == 0)
            walkingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorBlue, null));
        else if (idx == 1)
            upstairsTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorBlue, null));
        else if (idx == 2)
            downstairsTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorBlue, null));
        else if (idx == 3)
            standingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorBlue, null));
        else if (idx == 4)
            runningTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorBlue, null));
        else if (idx == 5)
            elevatorTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorBlue, null));
        else if (idx == 6)
            fallingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorBlue, null));
        else if (idx == 7)
            carTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorBlue, null));
    }*/

    private float[] toFloatArray(List<Float> list) {
        int i = 0;
        float[] array = new float[list.size()];

        for (Float f : list) {
            array[i++] = (f != null ? f : Float.NaN);
        }
        return array;
    }

    private static float round(float d, int decimalPlace) {
        BigDecimal bd = new BigDecimal(Float.toString(d));
        bd = bd.setScale(decimalPlace, BigDecimal.ROUND_HALF_UP);
        return bd.floatValue();
    }

    private SensorManager getSensorManager() {
        return (SensorManager) getSystemService(SENSOR_SERVICE);
    }

    public static Quaternion getOrientationVectorFromAccelerationMagnetic(float[] acceleration, float[] magnetic) {
        float[] rotationMatrix = new float[9];
        if (SensorManager.getRotationMatrix(rotationMatrix, null, acceleration, magnetic)) {
            double[] rotation = getQuaternion(new Matrix3f(rotationMatrix));
            return new Quaternion(rotation[0], rotation[1], rotation[2], rotation[3]);
        }

        return null;
    }

    private static double[] getQuaternion(Matrix3f m1) {
        double w = Math.sqrt(1.0 + m1.get(0,0) + m1.get(1,1) + m1.get(2,2)) / 2.0;
        double w4 = (4.0 * w);
        double x = (m1.get(2,1) - m1.get(1,2)) / w4 ;
        double y = (m1.get(0,2) - m1.get(2,0)) / w4 ;
        double z = (m1.get(1,0) - m1.get(0,1)) / w4 ;

        return new double[]{w,x,y,z};
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
}