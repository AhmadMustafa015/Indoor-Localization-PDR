package edu.onbasli.indoorlocalization.InertiaNavegation.activity;

import android.Manifest;
import android.annotation.TargetApi;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.material.floatingactionbutton.FloatingActionButton;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;

import edu.onbasli.indoorlocalization.InertiaNavegation.OrientationFusedKalman.LinearAccelerationFusion;
import edu.onbasli.indoorlocalization.InertiaNavegation.OrientationFusedKalman.OrientationFusedComplementary;
import edu.onbasli.indoorlocalization.InertiaNavegation.OrientationFusedKalman.OrientationFusedKalman;
import edu.onbasli.indoorlocalization.InertiaNavegation.OrientationFusedKalman.RotationUtil;
import edu.onbasli.indoorlocalization.InertiaNavegation.extra.KalmanFilterSimple;
import edu.onbasli.indoorlocalization.InertiaNavegation.filters.LowPassFilter;
import edu.onbasli.indoorlocalization.InertiaNavegation.filters.MeanFilter;
import edu.onbasli.indoorlocalization.InertiaNavegation.filters.MedianFilter;
import edu.onbasli.indoorlocalization.InertiaNavegation.stepcounting.StepDetection;
import edu.onbasli.indoorlocalization.R;
import edu.onbasli.indoorlocalization.InertiaNavegation.extra.ExtraFunctions;
import edu.onbasli.indoorlocalization.InertiaNavegation.filewriting.DataFileWriter;
import edu.onbasli.indoorlocalization.InertiaNavegation.graph.ScatterPlot;
import edu.onbasli.indoorlocalization.InertiaNavegation.orientation.GyroscopeDeltaOrientation;
import edu.onbasli.indoorlocalization.InertiaNavegation.orientation.GyroscopeEulerOrientation;
import edu.onbasli.indoorlocalization.InertiaNavegation.orientation.MagneticFieldOrientation;
import edu.onbasli.indoorlocalization.InertiaNavegation.prefs.PrefUtils;

public class GraphActivity extends AppCompatActivity implements SensorEventListener{
    private static final float GYROSCOPE_INTEGRATION_SENSITIVITY = 0.0025f;

    private static final String FOLDER_NAME = "Pedestrian_Dead_Reckoning/Graph_Activity";
    private static final String[] DATA_FILE_NAMES = {
            "Initial_Orientation",
            "Acceleration",
            "Gyroscope_Uncalibrated",
            "Magnetic_Field_Uncalibrated",
            "Gravity",
            "XY_Data_Set",
            "DebuggingHeading"
    };
    private static final String[] DATA_FILE_HEADINGS = {
            "Initial_Orientation",
            "Acceleration" + "\n" + "t,Ax,Ay,Az,stepState",
            "Gyroscope_Uncalibrated" + "\n" + "t,uGx,uGy,uGz,xBias,yBias,zBias,gyro_heading",
            "Magnetic_Field_Uncalibrated" + "\n" + "t,uMx,uMy,uMz,xBias,yBias,zBias,heading",
            "Gravity" + "\n" + "t,gx,gy,gz",
            "stepNumber,strideLength,totalDistance,time,Heading,originalPointX,originalPointY,rotatedPointX,rotatedPointY",
            "Time,RawGyroX,RawGyroY,RawGyroZ,xGBias,yGBias,zGBias,RawMgnX,RawMgnY,RawMgnZ,xMBias,yMBias,zMBias,HeadingGyro,HeadingMgn,HeadingComp,TotalSteps"
    };

    enum currentState {
        No_Steps,
        Step_Initial,
        Step_Terminate
    }
    private currentState stepState;
    private double[] quaternion;
    private float[] gravityCons = {0, 0, 9.81f};
    private float[] gravity;
    private StepDetection stepDetectionAlgorithm;
    private boolean isFirstRun;
    private double lastTimestamp;
    private boolean a0,a1,a2 = false;
    private double totalDistance = 0.0;
    private double lastStepLength;
    private static float thm = 0.5f;
    private static float thd = 0.3f;
    private long LSI = 0;
    private static int mind = 50;
    private long stepEpoch = 0;
    private double aMax;
    //private Vector aMin;
    private double tempMax = 0;
    private double tempMin = 999999f;
    private double K;
    private static double B = 0.7f;
    private int stepCounter;
    private TextView stepCounting;
    private TextView totalD;
    private GyroscopeDeltaOrientation gyroscopeDeltaOrientation;
    private GyroscopeEulerOrientation gyroscopeEulerOrientation;
    private DataFileWriter dataFileWriter;
    private ScatterPlot scatterPlot;

    private FloatingActionButton fabButton;
    private LinearLayout mLinearLayout;

    private SensorManager sensorManager;
    private Sensor sensorOrientation;
    int totalSteps;
    float[] gyroBias;
    float[] magBias;
    float[] accXBias;
    float[] accYBias;
    float[] accZBias;

    float[] currGravity; //current gravity
    float[] currMag; //current magnetic field
    float[][] gyroRotationMatrix;

    private boolean isRunning;
    private boolean isCalibrated;
    private boolean usingDefaultCounter;
    private boolean areFilesCreated;
    private float strideLength;
    private float gyroHeading;
    private float magHeading;
    private float kalmanHeading;

    private long startTime;
    private boolean firstRun;
    boolean useStatic = false;

    private float initialHeading;
    //Kalman
    private int window_size = 100; //number of readings before reset the Kalman filter state equations
    private int KF_itr = 0;
    private int KF_heading_itr = 0;
    private KalmanFilterSimple KF;
    private KalmanFilterSimple KF_heading;
    private boolean covearianceMatrixNotInitiated;
    private double[] Zobs;
    private boolean covarianceMatrixNotInitiated_heading;
    private double[] Zobs_heading;
    float[][] prev_orientation;
    double[][] _R;
    double[][] _R_2;
    double[][] magHeadingRotation;

    //New Age
    private float[] magnetic = new float[3];
    private float[] acceleration = new float[3];
    private float[] rotation = new float[3];
    private float[] RotationAngels = new float[3];  // RotationAngels[0] Heading
                                                    // RotationAngels[1] Pitch
                                                    //  RotationAngels[2] Roll
    private boolean hasAcceleration = false;
    private boolean hasMagnetic = false;
    protected Handler uiHandler;
    protected Runnable uiRunnable;
    private boolean stepDetected = false;
    private long accTime = 0;
    MeanFilter meanFilter;
    MeanFilter meanFilterAcc;
    MedianFilter medianFilter;
    LowPassFilter lowPassFilter;
    private OrientationFusedKalman orientationFusionKalman;
    private LinearAccelerationFusion linearAccelerationFilterKalman;
    private static float[] linearAcceleration = new float[3];

    private OrientationFusedComplementary orientationFusedComplementary;
    @TargetApi(Build.VERSION_CODES.KITKAT)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_graph);
        totalSteps = 0;
        gyroRotationMatrix = new float[3][3];

        // get location permission
        /*if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED
                || ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED
                || ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(GraphActivity.this, new String[] {
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
            },0);
            finish();
        }*/
        //step Detection
        totalD = (TextView) findViewById(R.id.TV_totalDistace);
        stepCounting = (TextView) findViewById(R.id.TV_numSteps);
        stepCounter = 0;
        stepState = currentState.No_Steps;
        quaternion = new double[4];
        gravity = new float[3];
        stepDetectionAlgorithm = new StepDetection();

        //defining needed variables
        gyroBias = null;
        magBias = null;
        accXBias = null;
        accYBias = null;
        accZBias = null;
        currGravity = null;
        currMag = null;

        String counterSensitivity;
        magHeadingRotation = new double[2][1];

        isRunning = isCalibrated = usingDefaultCounter = areFilesCreated = false;
        firstRun = true;
        strideLength = 0;
        initialHeading = gyroHeading = magHeading = 0;
        startTime = 0;

        //getting global settings
        strideLength = 0;
        //strideLength = strideLength / 3.281f;
        isCalibrated = getIntent().getBooleanExtra("is_calibrated", false);
        gyroBias = getIntent().getFloatArrayExtra("gyro_bias");
        magBias = getIntent().getFloatArrayExtra("mag_bias");
        accXBias = getIntent().getFloatArrayExtra("accX_bias");
        accYBias = getIntent().getFloatArrayExtra("accY_bias");
        accZBias = getIntent().getFloatArrayExtra("accZ_bias");
        Zobs = new double[]{ExtraFunctions.getVariance(accXBias), ExtraFunctions.getVariance(accYBias), ExtraFunctions.getVariance(accZBias)};
        Zobs_heading = new double[2];
        //using user_name to get index of user in userList, which is also the index of the user's stride_length
        counterSensitivity = getIntent().getStringExtra("preferred_step_counter");

        //usingDefaultCounter is counterSensitivity = "default" and sensor is available
        usingDefaultCounter = counterSensitivity.equals("default") &&
                getIntent().getBooleanExtra("step_detector", false);

        //initializing needed classes
        gyroscopeDeltaOrientation = new GyroscopeDeltaOrientation(GYROSCOPE_INTEGRATION_SENSITIVITY, gyroBias);

        //defining views
        fabButton = findViewById(R.id.fab);
        mLinearLayout = findViewById(R.id.linearLayoutGraph);

        //setting up graph with origin
        scatterPlot = new ScatterPlot("Gyro");
        scatterPlot.addPoint(0, 0); //AM: future update sit value using GPS
        //scatterPlot.addPointM(0, 0); //AM: future update sit value using GPS
        //scatterPlot.addPointC(0, 0); //AM: future update sit value using GPS
        mLinearLayout.addView(scatterPlot.getGraphView(getApplicationContext()));

        //message user w/ user_name and stride_length info
        //Toast.makeText(GraphActivity.this, "Stride Length: " + strideLength, Toast.LENGTH_SHORT).show();

        //starting

        //starting sensors
        //AM: Try fixed freq
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        sensorManager.registerListener(GraphActivity.this,
                sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY),
                SensorManager.SENSOR_DELAY_FASTEST);
        if(!PrefUtils.getPrefKalmanFilterAccEnabled(this)) {
            sensorOrientation = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);
            sensorManager.registerListener(GraphActivity.this, sensorOrientation, SensorManager.SENSOR_DELAY_FASTEST);
        }
        if (isCalibrated) {
            sensorManager.registerListener(GraphActivity.this,
                    sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED),
                    SensorManager.SENSOR_DELAY_FASTEST);
            sensorManager.registerListener(GraphActivity.this,
                    sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE_UNCALIBRATED),
                    SensorManager.SENSOR_DELAY_FASTEST);
            _R = new double[][]{{gyroBias[0],0,0},
                                {0,gyroBias[1],0},
                                {0,0,gyroBias[2]}};
            _R_2 = new double[][]{{gyroBias[1],0},
                                    {0,gyroBias[2]}};
        } else {
            sensorManager.registerListener(GraphActivity.this,
                    sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD),
                    SensorManager.SENSOR_DELAY_FASTEST);
            sensorManager.registerListener(GraphActivity.this,
                    sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE),
                    SensorManager.SENSOR_DELAY_FASTEST);
            _R = new double[][]{{0.1*0.1,0,0},
                                {0,0.1*0.1,0},
                                {0,0,0.1*0.1}};
            _R_2 = new double[][]{{0.1*0.1,0},
                                {0,0.1*0.1}};
        }
        //Step detector on acc
        sensorManager.registerListener(GraphActivity.this,
                sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER),
                SensorManager.SENSOR_DELAY_FASTEST);
        /*uiHandler = new Handler();
        uiRunnable = new Runnable() {
            @Override
            public void run() {
                uiHandler.postDelayed(this, 20);
                PlottingSteps(accTime);
            }};*/
        //setting up buttons
        fabButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                if (!isRunning) {

                    isRunning = true;

                    createFiles();

                    //AM: step1 get intial heading in repect to earth using magnetometer data
                    float[][] initialOrientation = MagneticFieldOrientation.getOrientationMatrix(currGravity, currMag, magBias);
                    initialHeading = MagneticFieldOrientation.getHeading(initialOrientation);
                    prev_orientation = initialOrientation;
                    //saving initial orientation data
                    dataFileWriter.writeToFile("Initial_Orientation", "init_Gravity: " + Arrays.toString(currGravity));
                    dataFileWriter.writeToFile("Initial_Orientation", "init_Mag: " + Arrays.toString(currMag));
                    dataFileWriter.writeToFile("Initial_Orientation", "mag_Bias: " + Arrays.toString(magBias));
                    dataFileWriter.writeToFile("Initial_Orientation", "gyro_Bias: " + Arrays.toString(gyroBias));
                    dataFileWriter.writeToFile("Initial_Orientation", "init_Orientation: " + Arrays.deepToString(initialOrientation));
                    dataFileWriter.writeToFile("Initial_Orientation", "init_Heading: " + initialHeading);

//                Log.d("init_heading", "" + initialHeading);

                    //TODO: fix rotation matrix
                    //gyroscopeEulerOrientation = new GyroscopeEulerOrientation(initialOrientation);

                    gyroscopeEulerOrientation = new GyroscopeEulerOrientation(ExtraFunctions.IDENTITY_MATRIX);

                    dataFileWriter.writeToFile("XY_Data_Set", "Initial_orientation: " +
                            Arrays.deepToString(initialOrientation));
                    dataFileWriter.writeToFile("Gyroscope_Uncalibrated", "Gyroscope_bias: " +
                            Arrays.toString(gyroBias));
                    dataFileWriter.writeToFile("Magnetic_Field_Uncalibrated", "Magnetic_field_bias:" +
                            Arrays.toString(magBias));

                    fabButton.setImageDrawable(ContextCompat.getDrawable(GraphActivity.this, R.drawable.ic_pause_black_24dp));

                } else {

                    fabButton.setImageDrawable(ContextCompat.getDrawable(GraphActivity.this, R.drawable.ic_play_arrow_black_24dp));

                }


            }
        });

        /*KF = new KalmanFilterSimple();
        KF_heading = new KalmanFilterSimple();
        KF.configure(3);
        KF_heading.configure(2);
        isFirstRun = true;
        covearianceMatrixNotInitiated = true;
        covarianceMatrixNotInitiated_heading = true;*/
        orientationFusionKalman = new OrientationFusedKalman();
        orientationFusedComplementary = new OrientationFusedComplementary();
        meanFilter = new MeanFilter();
        meanFilterAcc = new MeanFilter();
        medianFilter = new MedianFilter();
        lowPassFilter = new LowPassFilter();
        if(PrefUtils.getPrefMeanFilterSmoothingEnabled(this)){
            meanFilter.setTimeConstant(PrefUtils.getPrefMeanFilterSmoothingTimeConstant(this));
        }
        if(PrefUtils.getPrefFSensorComplimentaryLinearAccelerationEnabled(this)){
            orientationFusedComplementary.setTimeConstant(PrefUtils.getPrefFSensorComplimentaryLinearAccelerationTimeConstant(this));
        }
        if(PrefUtils.getPrefMeanFilterSmoothingEnabledAcc(this)){
            meanFilterAcc.setTimeConstant(PrefUtils.getPrefMeanFilterSmoothingTimeConstantAcc(this));
        }
        if(PrefUtils.getPrefMedianFilterSmoothingEnabled(this)){
            medianFilter.setTimeConstant(PrefUtils.getPrefMedianFilterSmoothingTimeConstant(this));
        }
        if(PrefUtils.getPrefLpfSmoothingEnabled(this)){
            lowPassFilter.setTimeConstant(PrefUtils.getPrefLpfSmoothingTimeConstant(this));
        }
        linearAccelerationFilterKalman = new LinearAccelerationFusion(orientationFusionKalman);

    }
    private void processAcceleration(float[] rawAcceleration, long acctime) {
        System.arraycopy(rawAcceleration, 0, this.acceleration, 0, this.acceleration.length);
        accTime = acctime;
    }
    private void processMagnetic(float[] magnetic) {
        System.arraycopy(magnetic, 0, this.magnetic, 0, this.magnetic.length);
    }

    private void processRotation(float[] rotation) {
        System.arraycopy(rotation, 0, this.rotation, 0, this.rotation.length);
    }
    private void processQuaternion(double[] _quaternion) {
        System.arraycopy(_quaternion, 0, this.quaternion, 0, this.quaternion.length);
    }

    @Override
    protected void onStop() {
        super.onStop();
        sensorManager.unregisterListener(this);
    }
    @Override
    protected void onPause() {
        super.onPause();
        if(PrefUtils.getPrefFSensorKalmanLinearAccelerationEnabled(this)) {
            orientationFusionKalman.stopFusion();
        }
        //uiHandler.removeCallbacksAndMessages(null);
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (isRunning) {

            // get location permission
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED
                    || ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED
                    || ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(GraphActivity.this, new String[] {
                        Manifest.permission.ACCESS_FINE_LOCATION,
                        Manifest.permission.ACCESS_COARSE_LOCATION,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE
                },0);
                finish();
            }

            if (isCalibrated) {
                sensorManager.registerListener(GraphActivity.this,
                        sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED),
                        SensorManager.SENSOR_DELAY_FASTEST);
                sensorManager.registerListener(GraphActivity.this,
                        sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE_UNCALIBRATED),
                        SensorManager.SENSOR_DELAY_FASTEST);
            } else {
                sensorManager.registerListener(GraphActivity.this,
                        sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD),
                        SensorManager.SENSOR_DELAY_FASTEST);
                sensorManager.registerListener(GraphActivity.this,
                        sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE),
                        SensorManager.SENSOR_DELAY_FASTEST);
            }

            if (usingDefaultCounter) {
                sensorManager.registerListener(GraphActivity.this,
                        sensorManager.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR),
                        SensorManager.SENSOR_DELAY_FASTEST);
            } else {
                sensorManager.registerListener(GraphActivity.this,
                        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER),
                        SensorManager.SENSOR_DELAY_FASTEST);
            }

            fabButton.setImageDrawable(ContextCompat.getDrawable(GraphActivity.this, R.drawable.ic_pause_black_24dp));

        } else {

            fabButton.setImageDrawable(ContextCompat.getDrawable(GraphActivity.this, R.drawable.ic_play_arrow_black_24dp));

        }

        if(PrefUtils.getPrefFSensorKalmanLinearAccelerationEnabled(GraphActivity.this)) {
            orientationFusionKalman.startFusion();
        }
       // uiHandler.post(uiRunnable);
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}

    @Override
    public void onSensorChanged(SensorEvent event) {

        if(firstRun) {
            startTime = event.timestamp;
            firstRun = false;
        }
        if (event.sensor.getType() == Sensor.TYPE_ROTATION_VECTOR && !PrefUtils.getPrefKalmanFilterAccEnabled(this))
        {
            float[] rotationVector = event.values;
            processQuaternion(stepDetectionAlgorithm.getQuaternion(rotationVector));
        }
        if (event.sensor.getType() == Sensor.TYPE_GRAVITY) {
            currGravity = event.values;
            //Log.d("gravity_values", Arrays.toString(event.values));
        } else if (event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD ||
                event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED) {
            currMag = event.values;

            processMagnetic(event.values);
            hasMagnetic = true;
//            Log.d("mag_values", Arrays.toString(event.values));
        }
        if (isRunning) {
            if (event.sensor.getType() == Sensor.TYPE_GRAVITY) {
                ArrayList<Float> dataValues = ExtraFunctions.arrayToList(event.values);
                dataValues.add(0, (float)(event.timestamp - startTime));
                dataFileWriter.writeToFile("Gravity", dataValues);
            } else if (event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD || event.sensor.getType() ==
                    Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED) {
                processMagnetic(event.values);
                hasMagnetic = true;


                float[][] orientationMatrix = MagneticFieldOrientation.getOrientationMatrix(currGravity, currMag, magBias);
                magHeading = MagneticFieldOrientation.getHeading(orientationMatrix); //yaw
                Zobs_heading[0] = orientationMatrix[0][0] - prev_orientation[0][0];
                Zobs_heading[1] = orientationMatrix[1][0] - prev_orientation[1][0];
                magHeadingRotation[0][0] = orientationMatrix[0][0];
                magHeadingRotation[1][0] = orientationMatrix[1][0];

//                Log.d("mag_heading", "" + magHeading);

                //saving magnetic field data
                ArrayList<Float> dataValues = ExtraFunctions.createList(
                        event.values[0], event.values[1], event.values[2],
                        magBias[0], magBias[1], magBias[2]
                );
                dataValues.add(0, (float)(event.timestamp - startTime));
                dataValues.add(magHeading);
                dataFileWriter.writeToFile("Magnetic_Field_Uncalibrated", dataValues);

            } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE ||
                    event.sensor.getType() == Sensor.TYPE_GYROSCOPE_UNCALIBRATED) {
                processRotation(event.values);

                if(PrefUtils.getPrefFSensorKalmanLinearAccelerationEnabled(this))
                {
                    if (!orientationFusionKalman.isBaseOrientationSet()) {
                        if (hasAcceleration && hasMagnetic) {
                            orientationFusionKalman.setBaseOrientation(RotationUtil.getOrientationVectorFromAccelerationMagnetic(acceleration, magnetic));
                        }
                    } else {
                        setOutput(orientationFusionKalman.calculateFusedOrientation(rotation, event.timestamp, acceleration, magnetic));
                        if(PrefUtils.getPrefKalmanFilterAccEnabled(this)) {
                            processLinearAcceleration(linearAccelerationFilterKalman.filter(acceleration));
                            processQuaternion(orientationFusionKalman.getQuaternion());
                        }
                        if(PrefUtils.getPrefMeanFilterSmoothingEnabled(this))
                        {
                            setOutput(
                                    meanFilter.filter(RotationAngels)
                            );
                        }
                        gyroHeading = RotationAngels[0];

                    }
                } else if(PrefUtils.getPrefFSensorComplimentaryLinearAccelerationEnabled(this)) {
                    if (!orientationFusedComplementary.isBaseOrientationSet()) {
                        if (hasAcceleration && hasMagnetic) {
                            orientationFusedComplementary.setBaseOrientation(RotationUtil.getOrientationVectorFromAccelerationMagnetic(acceleration, magnetic));
                        }
                    } else {
                        setOutput(orientationFusedComplementary.calculateFusedOrientation(rotation, event.timestamp, acceleration, magnetic));
                        if(PrefUtils.getPrefMeanFilterSmoothingEnabled(this))
                        {
                            setOutput(
                                    meanFilter.filter(RotationAngels)
                            );
                        }
                        gyroHeading = RotationAngels[0];
                    }

                } else if (!PrefUtils.getPrefFSensorComplimentaryLinearAccelerationEnabled(this) && !PrefUtils.getPrefFSensorKalmanLinearAccelerationEnabled(this))
                {
                    float[] deltaOrientation = gyroscopeDeltaOrientation.calcDeltaOrientation(event.timestamp, event.values);
                    //Kalman Filter: 1- initiate X matrix
                    gyroRotationMatrix = gyroscopeEulerOrientation.getOrientationMatrix(deltaOrientation);
                    double currentTime = ExtraFunctions.nsToSec(event.timestamp);
 /*                   double [][] X = ExtraFunctions.vectorToMatrix(new double[]{gyroRotationMatrix[2][0],gyroRotationMatrix[2][1],gyroRotationMatrix[2][2]});
                    if (isFirstRun) {
                        isFirstRun = false;
                        lastTimestamp = currentTime;
                        KF.setState(X);
                        KF_heading.setState(ExtraFunctions.vectorToMatrix(new double[]{gyroRotationMatrix[0][0],gyroRotationMatrix[1][0]}));

                    }
                    else {
                        if(covearianceMatrixNotInitiated)
                        {
                            KF.initiateErrorCovariance(new double[]{gyroRotationMatrix[2][0],gyroRotationMatrix[2][1],gyroRotationMatrix[2][2]});
                            KF_heading.initiateErrorCovariance(new double[]{gyroRotationMatrix[0][0],gyroRotationMatrix[1][0]});
                            covearianceMatrixNotInitiated = false;
                        }
                        KF.setState(X);
                        KF_heading.setState(ExtraFunctions.vectorToMatrix(new double[]{gyroRotationMatrix[0][0],gyroRotationMatrix[1][0]}));
                        double deltaT = currentTime - lastTimestamp;
                        double [][] B = {{0,-gyroRotationMatrix[2][2] * deltaT,gyroRotationMatrix[2][1] * deltaT},
                                {gyroRotationMatrix[2][2] * deltaT,0,-gyroRotationMatrix[2][0] * deltaT},
                                {-gyroRotationMatrix[2][1] * deltaT,gyroRotationMatrix[2][0] * deltaT,0}};
                        double [][] B_heading = {{-gyroRotationMatrix[0][2] * deltaT,gyroRotationMatrix[0][1] * deltaT},
                                {-gyroRotationMatrix[1][2] * deltaT,gyroRotationMatrix[1][1] * deltaT}};
                        KF_heading.setState(ExtraFunctions.vectorToMatrix(new double[]{gyroRotationMatrix[0][0],gyroRotationMatrix[1][0]}));
                        double [][] u = ExtraFunctions.vectorToMatrix(event.values);
                        double [][] u_heading = ExtraFunctions.vectorToMatrix(new double[]{event.values[1],event.values[2]});
                        KF.setControlMatrix(u,B);
                        KF_heading.setControlMatrix(u_heading,B_heading);
                        KF.predict();
                        KF_heading.predict();
                        KF.update(new double[]{gyroRotationMatrix[2][0],gyroRotationMatrix[2][1],gyroRotationMatrix[2][2]},Zobs,_R);
                        //KF_heading.update(new double[]{gyroRotationMatrix[0][0],gyroRotationMatrix[1][0]},Zobs_heading,_R_2);
                        KF_heading.update(magHeadingRotation,_R_2);
                           // new double[][]{{gyroBias[0],0,0},
                            //        {0,gyroBias[1],0},
                              //      {0,0,gyroBias[2]}});
                    }
*/
                    lastTimestamp = currentTime;
                    gyroHeading = gyroscopeEulerOrientation.getHeading(gyroRotationMatrix);
                    gyroHeading += initialHeading;

                }


//                Log.d("gyro_heading", "" + gyroHeading);

                //saving gyroscope data
                ArrayList<Float> dataValues = ExtraFunctions.createList(
                        event.values[0], event.values[1], event.values[2],
                        gyroBias[0], gyroBias[1], gyroBias[2]
                );
                //float[][] kH = KF_heading.getState();
                //kalmanHeading = (float) (Math.atan2(kH[1][0], kH[0][0])); //+ initialHeading);
                dataValues.add(0, (float)(event.timestamp - startTime));
                dataValues.add(gyroHeading);
                //dataValues.add(kalmanHeading);
                dataFileWriter.writeToFile("Gyroscope_Uncalibrated", dataValues);

            } else if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER)
            {
                if(PrefUtils.getPrefMeanFilterSmoothingEnabledAcc(this))
                    processAcceleration(meanFilterAcc.filter(event.values),event.timestamp);
                else if(PrefUtils.getPrefMedianFilterSmoothingEnabled(this))
                    processAcceleration(medianFilter.filter(event.values),event.timestamp);
                 else if(PrefUtils.getPrefLpfSmoothingEnabled(this))
                    processAcceleration(lowPassFilter.filter(event.values),event.timestamp);
                 else if (!PrefUtils.getPrefMeanFilterSmoothingEnabledAcc(this) && !PrefUtils.getPrefMedianFilterSmoothingEnabled(this) &&
                        !PrefUtils.getPrefLpfSmoothingEnabled(this))
                    processAcceleration(event.values, event.timestamp);
                hasAcceleration = true;

                float[] linearAcc = new float[3];
                if(PrefUtils.getPrefKalmanFilterAccEnabled(this))
                {
                    System.arraycopy(linearAcceleration, 0, linearAcc, 0, linearAcc.length);
                }
                else {
                    System.arraycopy(acceleration, 0, linearAcc, 0, linearAcc.length);
                    final float alpha = 0.8f;
                /*gravity[0] = alpha * gravity[0] + (1 - alpha) * sensorEvent.values[0];
                gravity[1] = alpha * gravity[1] + (1 - alpha) * sensorEvent.values[1];
                gravity[2] = alpha * gravity[2] + (1 - alpha) * sensorEvent.values[2];*/
                    float[][] bodyAcc = stepDetectionAlgorithm.GetBodyAcc(linearAcc, quaternion, gravityCons);
                    linearAcc[0] = bodyAcc[0][0];
                    linearAcc[1] = bodyAcc[1][0];
                    linearAcc[2] = bodyAcc[2][0];
                }
                float accMag = linearAcc[0] * linearAcc[0] + linearAcc[1] * linearAcc[1] + linearAcc[2] * linearAcc[2];
                accMag = (float) Math.sqrt(accMag);
                /*mSeries1.appendData(new DataPoint(stepEpoch, accMag),true, 40);
                mSeries2.appendData(new DataPoint(stepEpoch, linearAcc[2]),true, 40);
                mSeries3.appendData(new DataPoint(stepEpoch, thm),true, 40);
                */

                switch (stepState) {
                    case No_Steps:
                        if (accMag > thm && Math.abs(accMag - linearAcc[2]) < thd && (stepEpoch - LSI) > mind) {
                            stepDetected = true;
                            stepState = currentState.Step_Initial;
                        }
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
                            totalDistance = totalDistance + lastStepLength;
                            totalD.setText(String.format("%.2f",totalDistance));
                        }
                        ++stepEpoch;
                }
                if (stepDetected) {
                    stepDetected = false;
                    //AM: Draw here
                    //saving linear acceleration data
                    strideLength = (float) lastStepLength;
                    totalSteps++;
                    ArrayList<Float> dataValues = ExtraFunctions.arrayToList(acceleration);
                    dataValues.add(0, (float)(accTime - startTime));
                    dataValues.add(1f);
                    dataFileWriter.writeToFile("Acceleration", dataValues);

                    //complimentary filter
                    //float compHeading = ExtraFunctions.calcCompHeading(magHeading, gyroHeading);

                    //Log.d("comp_heading", "" + compHeading);

                    //getting and rotating the previous XY points so North 0 on unit circle
                    float oPointX = scatterPlot.getLastYPoint();
                    float oPointY = -scatterPlot.getLastXPoint();
                    /*float oPointXM = scatterPlot.getLastYPointM();
                    float oPointYM = -scatterPlot.getLastXPointM();
                    float oPointXC = scatterPlot.getLastYPointC();
                    float oPointYC = -scatterPlot.getLastXPointC();*/
                    //calculating XY points from heading and stride_length
                    //gyroHeading = (float) (Math.toDegrees(gyroHeading) + 360) % 360;
                    //gyroHeading = (float) (Math.toRadians(gyroHeading));
                    oPointX += ExtraFunctions.getXFromPolar(strideLength, gyroHeading);
                    oPointY += ExtraFunctions.getYFromPolar(strideLength, gyroHeading);

                    //rotating points by 90 degrees, so north is up
                    float rPointX = -oPointY;
                    float rPointY = oPointX;

                    scatterPlot.addPoint(rPointX, rPointY);
                    //scatterPlot.addPointM(rPointXM, rPointYM);
                    //scatterPlot.addPointM(rPointXC, rPointYC);
                    //saving XY location data
                    dataFileWriter.writeToFile("XY_Data_Set",
                            stepCounter,
                            strideLength,
                            (float)totalDistance,
                            (accTime - startTime),
                            //magHeading,
                            gyroHeading,
                            //kalmanHeading,
                            //compHeading,
                            oPointX,
                            oPointY,
                            rPointX,
                            rPointY);

                    mLinearLayout.removeAllViews();
                    mLinearLayout.addView(scatterPlot.getGraphView(getApplicationContext()));
                }else {
                    //saving acceleration data
                    ArrayList<Float> dataValues = ExtraFunctions.arrayToList(acceleration);
                    dataValues.add(0, (float) accTime);
                    if(stepState == currentState.No_Steps)
                        dataValues.add(0f);
                    else
                        dataValues.add(2f);
                    dataFileWriter.writeToFile("Acceleration", dataValues);
                }
            }
        }
    }

    private void createFiles() {
        if (!areFilesCreated) {
            try {
                dataFileWriter = new DataFileWriter(FOLDER_NAME, DATA_FILE_NAMES, DATA_FILE_HEADINGS);
            } catch (IOException e) {
                Log.e("GraphActivity", e.toString());
            }
            areFilesCreated = true;
        }
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode){
            case 0:
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(GraphActivity.this, "Thank you for providing permission!", Toast.LENGTH_SHORT).show();
                    finish();
                } else {
                    Toast.makeText(GraphActivity.this, "Need location permission to create tour.", Toast.LENGTH_LONG).show();
                    finish();
                }
                break;
        }
    }
    private void PlottingSteps (long time) {
        //if step is found, findStep == true
        if(!isRunning)
            return;
        float[] linearAcc = acceleration;
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
                /*mSeries1.appendData(new DataPoint(stepEpoch, accMag),true, 40);
                mSeries2.appendData(new DataPoint(stepEpoch, linearAcc[2]),true, 40);
                mSeries3.appendData(new DataPoint(stepEpoch, thm),true, 40);
                */

        switch (stepState) {
            case No_Steps:
                if (accMag > thm && Math.abs(accMag - linearAcc[2]) < thd && (stepEpoch - LSI) > mind) {
                    stepDetected = true;
                    stepState = currentState.Step_Initial;
                }
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
                    totalDistance = totalDistance + lastStepLength;
                    totalD.setText(String.format("%.2f",totalDistance));
                }
                ++stepEpoch;
        }
        if (stepDetected) {
            stepDetected = false;
            //AM: Draw here
            //saving linear acceleration data
            strideLength = (float) lastStepLength;
            totalSteps++;
            ArrayList<Float> dataValues = ExtraFunctions.arrayToList(acceleration);
            dataValues.add(0, (float)(time - startTime));
            dataValues.add(1f);
            dataFileWriter.writeToFile("Acceleration", dataValues);

            //complimentary filter
            //float compHeading = ExtraFunctions.calcCompHeading(magHeading, gyroHeading);

            //Log.d("comp_heading", "" + compHeading);

            //getting and rotating the previous XY points so North 0 on unit circle
            float oPointX = scatterPlot.getLastYPoint();
            float oPointY = -scatterPlot.getLastXPoint();
                    /*float oPointXM = scatterPlot.getLastYPointM();
                    float oPointYM = -scatterPlot.getLastXPointM();
                    float oPointXC = scatterPlot.getLastYPointC();
                    float oPointYC = -scatterPlot.getLastXPointC();*/
            //calculating XY points from heading and stride_length
        //gyroHeading = (float) (Math.toDegrees(gyroHeading) + 360) % 360;
        //gyroHeading = (float) (Math.toRadians(gyroHeading));
        oPointX += ExtraFunctions.getXFromPolar(strideLength, gyroHeading);
            oPointY += ExtraFunctions.getYFromPolar(strideLength, gyroHeading);

            //rotating points by 90 degrees, so north is up
            float rPointX = -oPointY;
            float rPointY = oPointX;

            scatterPlot.addPoint(rPointX, rPointY);
            //scatterPlot.addPointM(rPointXM, rPointYM);
            //scatterPlot.addPointM(rPointXC, rPointYC);
            //saving XY location data
            dataFileWriter.writeToFile("XY_Data_Set",
                    stepCounter,
                    strideLength,
                    (float)totalDistance,
                    (time - startTime),
                    //magHeading,
                    gyroHeading,
                    //kalmanHeading,
                    //compHeading,
                    oPointX,
                    oPointY,
                    rPointX,
                    rPointY);

            mLinearLayout.removeAllViews();
            mLinearLayout.addView(scatterPlot.getGraphView(getApplicationContext()));
        }else {
            //saving acceleration data
            ArrayList<Float> dataValues = ExtraFunctions.arrayToList(acceleration);
            dataValues.add(0, (float) time);
            if(stepState == currentState.No_Steps)
                dataValues.add(0f);
            else
                dataValues.add(2f);
            dataFileWriter.writeToFile("Acceleration", dataValues);
        }
    }
    private void setOutput(float[] value) {
        System.arraycopy(value, 0, RotationAngels, 0, value.length);
    }

    private void processLinearAcceleration(float[] linAcc) {
        System.arraycopy(linAcc, 0, this.linearAcceleration, 0, this.linearAcceleration.length);
    }

}

