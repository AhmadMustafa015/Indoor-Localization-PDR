package edu.onbasli.indoorlocalization.InertiaNavegation.filters;
import java.util.ArrayDeque;
import java.util.Arrays;

public class MeanFilter extends AveragingFilter {

    private static final String tag = MeanFilter.class.getSimpleName();

    private final ArrayDeque<float[]> values;
    private float[] output;

    /**
     * Initialize a new MeanFilter object.
     */
    public MeanFilter() {
        this(DEFAULT_TIME_CONSTANT);
    }

    public MeanFilter(float timeConstant) {
        super(timeConstant);
        values = new ArrayDeque<>();
    }

    /**
     * Filter the data.
     *
     * @param data contains input the data.
     * @return the filtered output data.
     */
    public float[] filter(float[] data) {

        if (startTime == 0) {
            startTime = System.nanoTime();
        }

        timestamp = System.nanoTime();

        // Find the sample period (between updates) and convert from
        // nanoseconds to seconds. Note that the sensor delivery rates can
        // individually vary by a relatively large time frame, so we use an
        // averaging technique with the number of sensor updates to
        // determine the delivery rate.
        float hz = (count++ / ((timestamp - startTime) / 1000000000.0f));

        int filterWindow = (int) Math.ceil(hz * timeConstant);

        values.addLast(Arrays.copyOf(data, data.length));

        while (values.size() > filterWindow) {
            values.removeFirst();
        }

        if(!values.isEmpty()) {
            output = getMean(values);
        } else {
            output = new float[data.length];
            System.arraycopy(data, 0, output, 0, data.length);
        }

        return output;
    }

    @Override
    public float[] getOutput() {
        return output;
    }

    /**
     * Get the mean of the data set.
     *
     * @param data the data set.
     * @return the mean of the data set.
     */
    private float[] getMean(ArrayDeque<float[]> data) {
        float[] mean = new float[data.getFirst().length];

        for (float[] axis : data) {
            for (int i = 0; i < axis.length; i++) {
                mean[i] += axis[i];
            }
        }

        for (int i = 0; i < mean.length; i++) {
            mean[i] /= data.size();
        }

        return mean;
    }

    public void setTimeConstant(float timeConstant) {
        this.timeConstant = timeConstant;
    }

    public void reset() {
        super.reset();

        if(values != null) {
            this.values.clear();
        }
    }
}
