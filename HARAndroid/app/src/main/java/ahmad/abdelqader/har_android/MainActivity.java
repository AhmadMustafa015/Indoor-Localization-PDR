package ahmad.abdelqader.har_android;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.res.ResourcesCompat;

import android.util.Log;
import android.widget.TableRow;
import android.widget.TextView;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;



public class MainActivity extends AppCompatActivity implements SensorEventListener, TextToSpeech.OnInitListener {


    private static final int N_SAMPLES = 128;
    private static int prevIdx = -1;

    private static List<Float> ax;
    private static List<Float> ay;
    private static List<Float> az;

    private static List<Float> lx;
    private static List<Float> ly;
    private static List<Float> lz;

    private static List<Float> gx;
    private static List<Float> gy;
    private static List<Float> gz;

    private static List<Float> ma;
    private static List<Float> ml;
    private static List<Float> mg;
    private SensorManager mSensorManager;
    private Sensor mAccelerometer;
    private Sensor mGyroscope;


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

    private TableRow downstairsTableRow;
    private TableRow runningTableRow;
    private TableRow standingTableRow;
    private TableRow upstairsTableRow;
    private TableRow walkingTableRow;
    private TableRow elevatorTableRow;
    private TableRow fallingTableRow;
    private TableRow carTableRow;
    private TextToSpeech textToSpeech;

    private float[] results;
    private HARClassifier classifier;
    private static final float Frequency = 50f;
    private static final float nyquestRate = Frequency/2;
    private static final float cutoff_low = 0.3f;
    private static final float cutoff_high = 20f;
    private String[] labels = {"Walking","Upstairs","Downstairs", "Standing_Sitting", "Running","Elevator","Falling","Car"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ax = new ArrayList<>(); ay = new ArrayList<>(); az = new ArrayList<>();
        lx = new ArrayList<>(); ly = new ArrayList<>(); lz = new ArrayList<>();
        gx = new ArrayList<>(); gy = new ArrayList<>(); gz = new ArrayList<>();
        ma = new ArrayList<>(); ml = new ArrayList<>(); mg = new ArrayList<>();
        downstairsTextView = findViewById(R.id.downstairs_prob);
        runningTextView = findViewById(R.id.jogging_prob);
        standingTextView = findViewById(R.id.standing_prob);
        upstairsTextView = findViewById(R.id.upstairs_prob);
        walkingTextView = findViewById(R.id.walking_prob);
        elevatorTextView = findViewById(R.id.elevator_prob);
        fallingTextView = findViewById(R.id.falling_prob);
        carTextView = findViewById(R.id.car_prob);


        accX = findViewById(R.id.totalAcc_x);
        accY = findViewById(R.id.totalAcc_y);
        accZ = findViewById(R.id.totalAcc_z);
        linAccX = findViewById(R.id.linearAcc_x);
        linAccY = findViewById(R.id.linearAcc_y);
        linAccZ = findViewById(R.id.linearAcc_z);
        gyroX = findViewById(R.id.gyro_x);
        gyroY = findViewById(R.id.gyro_y);
        gyroZ = findViewById(R.id.gyro_z);

        downstairsTableRow = findViewById(R.id.downstairs_row);
        runningTableRow = findViewById(R.id.jogging_row);
        standingTableRow = findViewById(R.id.standing_row);
        upstairsTableRow = findViewById(R.id.upstairs_row);
        walkingTableRow = findViewById(R.id.walking_row);
        elevatorTableRow = findViewById(R.id.elevator_row);
        fallingTableRow = findViewById(R.id.elevator_row);
        carTableRow = findViewById(R.id.elevator_row);


        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mSensorManager.registerListener(this, mAccelerometer , SensorManager.SENSOR_DELAY_FASTEST);

        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mSensorManager.registerListener(this, mGyroscope , SensorManager.SENSOR_DELAY_FASTEST);

        classifier = new HARClassifier(getApplicationContext());

        textToSpeech = new TextToSpeech(this, this);
        textToSpeech.setLanguage(Locale.US);
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

                if(max > 0.50 && idx != prevIdx) {
                    textToSpeech.speak(labels[idx], TextToSpeech.QUEUE_ADD, null,
                            Integer.toString(new Random().nextInt()));
                    prevIdx = idx;
                }
            }
        }, 1000, 3000);
    }

    protected void onResume() {
        super.onResume();
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_FASTEST );
        //getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION), SensorManager.SENSOR_DELAY_FASTEST );
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_GYROSCOPE), SensorManager.SENSOR_DELAY_FASTEST );
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
            accX.setText(Float.toString(round(event.values[0], 4)));
            accY.setText(Float.toString(round(event.values[1], 4)));
            accZ.setText(Float.toString(round(event.values[2], 4)));
        }
         else if (sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            gx.add(event.values[0]);
            gy.add(event.values[1]);
            gz.add(event.values[2]);
            gyroX.setText(Float.toString(round(event.values[0], 4)));
            gyroY.setText(Float.toString(round(event.values[1], 4)));
            gyroZ.setText(Float.toString(round(event.values[2], 4)));
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    private void activityPrediction() {

        float [][][] inputData = new float[1][128][12];
        if (ax.size() >= N_SAMPLES && ay.size() >= N_SAMPLES && az.size() >= N_SAMPLES
               // && lx.size() >= N_SAMPLES && ly.size() >= N_SAMPLES && lz.size() >= N_SAMPLES
                && gx.size() >= N_SAMPLES && gy.size() >= N_SAMPLES && gz.size() >= N_SAMPLES
        ) {
            float[] axM = medianFilter(ax,3);
            double maValue; double mgValue; double mlValue;
            float[] accX = new float[N_SAMPLES];
            float[] bodyAccX = new float[N_SAMPLES];
            FilterAndExtraction(axM,cutoff_low,cutoff_high,accX,bodyAccX);
            float[] ayM = medianFilter(ay,3);
            float[] accY = new float[N_SAMPLES];
            float[] bodyAccY = new float[N_SAMPLES];
            FilterAndExtraction(ayM,cutoff_low,cutoff_high,accY,bodyAccY);
            float[] azM = medianFilter(az,3);
            float[] accZ = new float[N_SAMPLES];
            float[] bodyAccZ = new float[N_SAMPLES];
            FilterAndExtraction(azM,cutoff_low,cutoff_high,accZ,bodyAccZ);

            float[] gxM = medianFilter(gx,3);
            float[] bodyGyroX = new float[N_SAMPLES];
            float[] emptyX = new float[N_SAMPLES];
            FilterAndExtraction(gxM,cutoff_low,cutoff_high,emptyX,bodyGyroX);
            float[] gyM = medianFilter(gy,3);
            float[] bodyGyroY = new float[N_SAMPLES];
            float[] emptyY = new float[N_SAMPLES];
            FilterAndExtraction(gyM,cutoff_low,cutoff_high,emptyY,bodyGyroY);
            float[] gzM = medianFilter(gz,3);
            float[] bodyGyroZ = new float[N_SAMPLES];
            float[] emptyZ = new float[N_SAMPLES];
            FilterAndExtraction(gzM,cutoff_low,cutoff_high,emptyZ,bodyGyroZ);
            for( int i = 0; i < N_SAMPLES ; i++ ) {
                maValue = Math.sqrt(Math.pow(accX[i], 2) + Math.pow(accY[i], 2) + Math.pow(accZ[i], 2));
                mlValue = Math.sqrt(Math.pow(bodyAccX[i], 2) + Math.pow(bodyAccY[i], 2) + Math.pow(bodyAccZ[i], 2));
                mgValue = Math.sqrt(Math.pow(bodyGyroX[i], 2) + Math.pow(bodyGyroY[i], 2) + Math.pow(bodyGyroZ[i], 2));

                ma.add((float)maValue);
                ml.add((float)mlValue);
                mg.add((float)mgValue);
            }

            for (int i = 0; i < 128; i++)
            {
                inputData[0][i][0] = accX[i];
                inputData[0][i][1] = accY[i];
                inputData[0][i][2] = accZ[i];

                inputData[0][i][3] = bodyAccX[i];
                inputData[0][i][4] = bodyAccY[i];
                inputData[0][i][5] = bodyAccZ[i];

                inputData[0][i][6] = bodyGyroX[i];
                inputData[0][i][7] = bodyGyroY[i];
                inputData[0][i][8] = bodyGyroZ[i];

                inputData[0][i][9] = ma.get(i);
                inputData[0][i][10] = ml.get(i);
                inputData[0][i][11] = mg.get(i);
            }
            results = classifier.predictProbabilities(inputData);
            float max = -1;
            int idx = -1;
            for (int i = 0; i < results.length; i++)
            {
                if (results[i] > max) {
                    idx = i;
                    max = results[i];
                }
            }

            setProbabilities();
            setRowsColor(idx);

            ax.clear();
            ay.clear();
            az.clear();
            gx.clear();
            gy.clear();
            gz.clear();
            ma.clear();
            ml.clear();
            mg.clear();


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

    private void setRowsColor(int idx) {
        downstairsTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        runningTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        standingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        upstairsTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        walkingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        elevatorTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        fallingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        carTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));

        if(idx == 0)
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
    }

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
    private void FilterAndExtraction(float[] signalT, float DCcutoff, float HighCutoff,float[] totalAcc,float[] bodyAcc) {
        int length = N_SAMPLES;
        Log.d("array size: ", String.valueOf(signalT.length));
        Complex[] tSignal = new Complex[length];
        FFT ffT = new FFT();
        for (int i = 0; i < length; i++)
        {
            tSignal[i] = new Complex(signalT[i],0);
        }
        Complex[] signalF = FFT.fft(tSignal);
        float sampling_freq = Frequency/(float)length;
        float[] freq_range = new float[length];
        Complex[] noiseF = new Complex[length];
        Complex[] bodyF = new Complex[length];
        for (int i = 0; i < length;i++)
        {
            if(sampling_freq * i <= nyquestRate)
                freq_range[i] = sampling_freq * i;
            else
                freq_range[i] = -1 * freq_range[length - i];
            if(Math.abs(freq_range[i]) <= DCcutoff || Math.abs(freq_range[i]) > HighCutoff)
            {
                bodyF[i] = new Complex(0,0);
            }
            else
                bodyF[i] = signalF[i];
            if(Math.abs(freq_range[i]) <= HighCutoff)
                noiseF[i] = new Complex(0,0);
            else
                noiseF[i] = signalF[i];
        }
        float[] noiseSignalT = new float[length];
        Complex[] bodyT = FFT.ifft(bodyF);
        Complex[] noiseT = FFT.ifft(noiseF);
        for (int i = 0; i < length;i++)
        {
            bodyAcc[i] =(float) bodyT[i].re();
            noiseSignalT[i] =(float) noiseT[i].re();
            totalAcc[i] = signalT[i] - noiseSignalT[i];
        }
    }
    private static float[] medianFilter(List<Float> medFilter, int k)
    {
        int k2 = (k-1)/2;
        float[][] y = new float[medFilter.size()][k];
        for(int itr = 0; itr<medFilter.size();itr++)
            y[itr][k2] = medFilter.get(itr);
        for(int i = 0; i < k2;i++)
        {
            int j = k2-i;
            for(int index = 0; index<medFilter.size()-j; index++)
                y[j+index][i] = medFilter.get(index);

            for(int index = 0; index<j; index++)
                y[index][i] = medFilter.get(0);

            for(int index = 0; index<medFilter.size()-j; index++)
                y[index][k-(i+1)] = medFilter.get(index+j);

            for(int index = medFilter.size()-j; index<medFilter.size(); index++)
                y[index][k-(i+1)] = medFilter.get(medFilter.size()-1);



        }
        float[] filterdian = new float[medFilter.size()];
        for (int ii = 0; ii < medFilter.size();ii++)
        {
            filterdian[ii] = median(y[ii]);
        }


        return filterdian;
    }
    static float median(float[] values) {
        // sort array
        Arrays.sort(values);
        float medianVal;
        // get count of scores
        int totalElements = values.length;
        // check if total number of scores is even
        if (totalElements % 2 == 0) {
            float sumOfMiddleElements = values[totalElements / 2] +
                    values[totalElements / 2 - 1];
            // calculate average of middle elements
            medianVal =  sumOfMiddleElements / 2;
        } else {
            // get the middle element
            medianVal = values[values.length / 2];
        }
        return medianVal;
    }
}
