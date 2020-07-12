package com.example.stepdetectionandstepestimation;
import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.view.Gravity;
import android.view.Menu;
import android.view.MenuItem;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import com.jjoe64.graphview.GraphView;
import androidx.core.content.ContextCompat;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;
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
    private Sensor sensorOrientation;
    private Sensor sensorGravity;
    private SensorManager sensorManager;
    private int stepCounter = 0;

    //Declaring views
    private Button buttonStopCounter;
    private TextView stepCounting;
    private Button fabButton;
    private GraphView graph;
    boolean isRunning = false;

    private float[] quaternion;
    private float[] gravityCons = {0, 0, 9.81f};
    private float[] gravity;
    private StepDetection stepDetectionAlgorithm;

    private boolean a0,a1,a2 = false;
    private static float thm = 1.5f;
    private static float thd = 0.5f;
    private long LSI = 0;
    private static int mind = 50;
    private long stepEpoch = 0;
    private Vector aMax;
    private Vector aMin;
    private float tempMax = 0;
    private float tempMin = 999999f;
    private LineGraphSeries<DataPoint> mSeries1;
    private LineGraphSeries<DataPoint> mSeries2;
    private LineGraphSeries<DataPoint> mSeries3;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.fragment_first);
        //defining snesors
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        sensorAccelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        sensorLinearAcceleration = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        sensorMagnaticField = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        sensorOrientation = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);

        aMax = new Vector();
        aMin = new Vector();
        stepState = currentState.No_Steps;
        quaternion = new float[4];
        gravity = new float[3];
        stepDetectionAlgorithm = new StepDetection();
        fabButton = (Button) findViewById(R.id.button_start);
        graph = (GraphView) findViewById(R.id.graph);
        buttonStopCounter = (Button) findViewById(R.id.button_stop);
        stepCounting = (TextView) findViewById(R.id.textview_numSteps);
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
                sensorManager.unregisterListener(SensorActivity.this, sensorLinearAcceleration);
                sensorManager.unregisterListener(SensorActivity.this, sensorOrientation);
                sensorManager.unregisterListener(SensorActivity.this, sensorMagnaticField);
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
                    sensorManager.registerListener(SensorActivity.this, sensorLinearAcceleration, SensorManager.SENSOR_DELAY_FASTEST);
                    sensorManager.registerListener(SensorActivity.this, sensorMagnaticField, SensorManager.SENSOR_DELAY_FASTEST);
                    sensorManager.registerListener(SensorActivity.this, sensorOrientation, SensorManager.SENSOR_DELAY_FASTEST);
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
                        aMax.add(tempMax);

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
                        aMin.add(tempMin);
                    }
                    ++stepEpoch;
            }
        }
        if (sensorEvent.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {

        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
    @Override
    protected void onResume(){
        super.onResume();
        if(isRunning){
            sensorManager.registerListener(SensorActivity.this,
                    sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION),sensorManager.SENSOR_DELAY_FASTEST);
            sensorManager.registerListener(SensorActivity.this,
                    sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_FASTEST);
            sensorManager.registerListener(SensorActivity.this,
                    sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD), SensorManager.SENSOR_DELAY_FASTEST);
            sensorManager.registerListener(SensorActivity.this,
                    sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR), SensorManager.SENSOR_DELAY_FASTEST);
        }
    }
}
