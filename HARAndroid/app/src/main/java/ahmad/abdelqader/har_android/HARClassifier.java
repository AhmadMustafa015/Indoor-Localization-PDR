package ahmad.abdelqader.har_android;

import android.content.Context;
import android.content.res.AssetFileDescriptor;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class HARClassifier {
    Interpreter tflite;
    private static final int OUTPUT_SIZE = 8;

    public HARClassifier(final Context context) {
            try {
                tflite = new Interpreter(loadModelFile(context));
            }catch (Exception ex){
                ex.printStackTrace();
            }
        }
    private MappedByteBuffer loadModelFile(final Context context) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd("RNN_8.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    public float[] predictProbabilities(float[][][] data) {
        float[][] result_raw = new float[1][OUTPUT_SIZE];
        tflite.run(data,result_raw);
        float[] result = result_raw[0];
        //Log.d("this is my array", "arr: " + Arrays.toString(result));
        //Log.d("Input", "arr: " + Arrays.toString(data));
        //Walking       Upstairs     Downstairs 	Standing_Standing	Running
        return result;
    }
}
