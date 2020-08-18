package com.example.stepdetectionandstepestimation;
import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import com.jjoe64.graphview.GraphView;

import com.jjoe64.graphview.LegendRenderer;
import com.jjoe64.graphview.series.DataPoint;
import com.jjoe64.graphview.series.LineGraphSeries;

import java.util.Vector;

public class SensorActivity extends Activity implements SensorEventListener {
    public SensorActivity() {
    }

    enum currentState {
        No_Steps,
        Step_Initial,
        Step_Terminate
    }
    private currentState stepState;
    //Declaring Sensors
    private Sensor sensorAccelerometer;
    private Sensor sensorLinearAcceleration;
    private Sensor sensorMagnaticField;
    private Sensor sensorPressure;
    private Sensor sensorOrientation;
    private Sensor sensorGravity;
    private SensorManager sensorManager;
    private int stepCounter = 0;

    //Declaring views
    private Button buttonStopCounter;
    private TextView stepCounting;
    private TextView lastStepD;
    private TextView totalD;
    private TextView hight_p;

    private Button fabButton;
    private GraphView graph;
    boolean isRunning = false;

    private float[] quaternion;
    private float[] gravityCons = {0, 0, 9.81f};
    private float[] gravity;
    private StepDetection stepDetectionAlgorithm;

    private boolean a0,a1,a2 = false;
    private double totalDistance = 0.0;
    private double lastStepLength;
    private static float thm = 1.5f;
    private static float thd = 0.5f;
    private long LSI = 0;
    private static int mind = 50;
    private long stepEpoch = 0;
    private double aMax;
    //private Vector aMin;
    private double tempMax = 0;
    private double tempMin = 999999f;
    private double K;
    private static double B = 0.7f;
    private LineGraphSeries<DataPoint> mSeries1;
    private LineGraphSeries<DataPoint> mSeries2;
    private LineGraphSeries<DataPoint> mSeries3;
    //height estimation
   // private final int timeToUpdate = 15; //time which after we will see if the floor number changed
    private final int timeToAverage = 5; //Take avg readings every 1s
    private Vector<Float> avgReadings;
    private float avgT_2 = 0; //average values before 3 seconds;
    private float avgT_0 = 0; //average values in the last 1 seconds;
    private float avgT_5 = 0; //average values in the last 5 seconds;
    private float pstart = 0; //p at start walking up or down;
    private float pend = 0; //p at the end of walking up or down;

    private int currentFloor = 0;
   // private float lastAverageTime, lastUpdateTime = 0;
    private final float mPh_btwFloors = 0.30f;
    private boolean first_run = true;
    private boolean firstRun = true;
    private TextView floorNum;
    private float currentTime = 0;
    private final float thetaT = 0.1f; //hpa
    private int N0 , N1 = 5; //to remove ping-pong effect
    private int num1, num2 = 0; // counters
    private boolean enterLoop2 = false;
    private int upORdown = 0; //1 means up 2 means down
    private final float h0 = 3.52f; //height of each layer in the building
    private final float temperature = 27.0f;
    private final float segma = 1/273.15f;
    private final float eqConst = 18410.183f;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.fragment_first);
        //Height
        avgReadings = new Vector<Float>();
        //defining snesors
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        sensorAccelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        sensorLinearAcceleration = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        sensorMagnaticField = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        sensorOrientation = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);
        sensorPressure = sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE);
        //aMax = new Vector();
        //aMin = new Vector();
        stepState = currentState.No_Steps;
        quaternion = new float[4];
        gravity = new float[3];
        stepDetectionAlgorithm = new StepDetection();
        fabButton = (Button) findViewById(R.id.button_start);
        graph = (GraphView) findViewById(R.id.graph);
        buttonStopCounter = (Button) findViewById(R.id.button_stop);
        stepCounting = (TextView) findViewById(R.id.textview_numSteps);
        lastStepD = (TextView) findViewById(R.id.textview_lastStepD);
        totalD = (TextView) findViewById(R.id.textview_totalD);
        hight_p = (TextView) findViewById(R.id.textview_hight);
        floorNum = (TextView) findViewById(R.id.textview_floor);

        mSeries1 = new LineGraphSeries<>();
        mSeries2 = new LineGraphSeries<>();
        mSeries3 = new LineGraphSeries<>();
        mSeries2.setColor(Color.RED);
        mSeries3.setColor(Color.GRAY);
        mSeries3.setDrawDataPoints(true);
        mSeries3.setDataPointsRadius(10);
        mSeries1.setTitle("Acc_Mag");
        mSeries2.setTitle("Acc_Z");
        mSeries3.setTitle("thm");
        graph.addSeries(mSeries2);
        graph.addSeries(mSeries1);
        graph.addSeries(mSeries3);
        graph.getViewport().setXAxisBoundsManual(true);
        graph.getViewport().setMinX(0);
        graph.getViewport().setMaxX(40);
        graph.getLegendRenderer().setVisible(true);
        graph.getLegendRenderer().setAlign(LegendRenderer.LegendAlign.TOP);
        buttonStopCounter.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                sensorManager.unregisterListener(SensorActivity.this, sensorAccelerometer);
                //sensorManager.unregisterListener(SensorActivity.this, sensorLinearAcceleration);
                sensorManager.unregisterListener(SensorActivity.this, sensorOrientation);
               // sensorManager.unregisterListener(SensorActivity.this, sensorMagnaticField);
                buttonStopCounter.setEnabled(false);
                fabButton.setEnabled(true);
                isRunning = false;
                stepCounting.setText("0");
                stepCounter = 0;
            }
        });
        fabButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (!isRunning) {
                    isRunning = true;
                    sensorManager.registerListener(SensorActivity.this, sensorAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
                    //sensorManager.registerListener(SensorActivity.this, sensorLinearAcceleration, SensorManager.SENSOR_DELAY_FASTEST);
                    //sensorManager.registerListener(SensorActivity.this, sensorMagnaticField, SensorManager.SENSOR_DELAY_FASTEST);
                    sensorManager.registerListener(SensorActivity.this, sensorOrientation, SensorManager.SENSOR_DELAY_FASTEST);
                    sensorManager.registerListener(SensorActivity.this, sensorPressure, SensorManager.SENSOR_DELAY_NORMAL);
                    fabButton.setEnabled(false);
                    buttonStopCounter.setEnabled(true);
                }

            }
        });
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        if (sensorEvent.sensor.getType() == Sensor.TYPE_ROTATION_VECTOR)
        {
            float[] rotationVector = sensorEvent.values;
            quaternion = stepDetectionAlgorithm.getQuaternion(rotationVector);
        }
        if (sensorEvent.sensor.getType() == Sensor.TYPE_ACCELEROMETER)
        {
            if(quaternion  == null)
                return;
            float[] linearAcc = sensorEvent.values;
            final float alpha = 0.8f;
            /*gravity[0] = alpha * gravity[0] + (1 - alpha) * sensorEvent.values[0];
            gravity[1] = alpha * gravity[1] + (1 - alpha) * sensorEvent.values[1];
            gravity[2] = alpha * gravity[2] + (1 - alpha) * sensorEvent.values[2];*/
            float[][] bodyAcc = stepDetectionAlgorithm.GetBodyAcc(linearAcc, quaternion, gravityCons);
            linearAcc[0] = bodyAcc[0][0];
            linearAcc[1] = bodyAcc[1][0];
            linearAcc[2] = bodyAcc[2][0];
            float accMag = linearAcc[0] * linearAcc[0] + linearAcc[1] * linearAcc[1] + linearAcc[2] * linearAcc[2];
            accMag = (float) Math.sqrt(accMag);
            mSeries1.appendData(new DataPoint(stepEpoch, accMag),true, 40);
            mSeries2.appendData(new DataPoint(stepEpoch, linearAcc[2]),true, 40);
            mSeries3.appendData(new DataPoint(stepEpoch, thm),true, 40);
            switch (stepState) {
                case No_Steps:
                    if (accMag > thm && Math.abs(accMag - linearAcc[2]) < thd && (stepEpoch - LSI) > mind)
                        stepState = currentState.Step_Initial;
                    ++stepEpoch;
                    break;
                case Step_Initial:
                    if(linearAcc[2] > tempMax) {
                        tempMax = linearAcc[2];
                        LSI = stepEpoch;
                    }
                    if(linearAcc[2] < 0) {
                        stepState = currentState.Step_Terminate;
                        ++stepCounter;
                        stepCounting.setText(String.valueOf(stepCounter));
                        //aMax.add(tempMax);
                        aMax = tempMax;
                        K = (float) (B * (1.0/Math.pow(tempMax,0.33333333333)));
                    }
                    ++stepEpoch;
                    break;
                case Step_Terminate:
                    if(linearAcc[2] < tempMin) {
                        tempMin = linearAcc[2];
                    }
                    if (linearAcc[2] > thm)
                    {
                        stepState = currentState.No_Steps;
                        //aMin.add(tempMin);
                        lastStepLength = K * (Math.pow(aMax - tempMin, 0.25));
                        //lastStepD.setText(String.format("%.2f",lastStepLength));
                        totalDistance = totalDistance + lastStepLength;
                        //totalD.setText(String.format("%.2f",totalDistance));
                    }
                    ++stepEpoch;
            }
        }
        if (sensorEvent.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {

        }
        if (sensorEvent.sensor.getType() == Sensor.TYPE_PRESSURE)
        {
            if(sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE) == null)
            {
                hight_p.setText("NaN");
                floorNum.setText("NaN");
                return;
            }
            //final float deltaT = currentTime - lastAverageTime;
            //final float deltaTU = currentTime - lastUpdateTime;
            if(avgReadings.size() < timeToAverage)
            {
                avgReadings.add(sensorEvent.values[0]); //pt
            }
            else {
                avgT_0 = 0;
                for (float value :avgReadings) {
                    avgT_0 += value;
                }
                avgT_0 /= (float) avgReadings.size(); //pt average
                if(firstRun)
                {
                    avgT_2 = avgT_0;
                    firstRun = false;
                }
                avgReadings.clear();
                if(currentTime == 2)
                {
                    if (!enterLoop2 ) {
                        if (Math.abs(avgT_0 - avgT_2) > thetaT) {
                            if (num1 == 0)
                                avgT_5 = avgT_0;
                            if (num1 == N0) {
                                pstart = avgT_5;
                                enterLoop2 = true;
                                num1 = 0;
                                lastStepD.setText(String.format("%.2f", pstart));
                            }
                            else
                                ++num1;
                        } else {
                            num1 = 0;
                        }
                    }else {
                        if (Math.abs(avgT_0 - avgT_2) < thetaT){
                            if (num2 == 0)
                                avgT_5 = avgT_0;
                            if (num2 == N1) {
                                pend = avgT_5;
                                enterLoop2 = false;
                                num2 = 0;
                                currentFloor += Math.round((1/h0) * eqConst * (1+segma * temperature) * Math.log10(pstart/pend));
                                floorNum.setText(String.valueOf(currentFloor));
                                totalD.setText(String.format("%.2f", pend));
                            }
                            else
                                ++num2;
                        }else {
                            num2 = 0;
                        }
                    }
                    avgT_2 = avgT_0;
                    currentTime = 0;
                }
                ++currentTime;
            }
                hight_p.setText(String.format("%.2f", sensorEvent.values[0]));

        }

    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
    @Override
    protected void onResume(){
        super.onResume();
        if(isRunning){
        //    sensorManager.registerListener(SensorActivity.this,
        //            sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION),sensorManager.SENSOR_DELAY_FASTEST);
            sensorManager.registerListener(SensorActivity.this,
                    sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_FASTEST);
            //sensorManager.registerListener(SensorActivity.this,
            //        sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD), SensorManager.SENSOR_DELAY_FASTEST);
            sensorManager.registerListener(SensorActivity.this,
                    sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR), SensorManager.SENSOR_DELAY_FASTEST);
        }
    }
}
