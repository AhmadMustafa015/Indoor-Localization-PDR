package ahmad.abdelqader.navigation_deeplearning;

import android.content.Context;
import android.content.res.AssetFileDescriptor;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

public class HARClassifier {
    Interpreter tflite;
    private static final int OUTPUT_SIZE = 8;
    Map<Integer, Object> outputs = new HashMap<>();
    float[][] parsedOutput0 = new float[1][4];
    float[][] parsedOutput1 = new float[1][3];
    //private static final String MULTIPLE_INPUTS_MODEL_PATH = "tensorflow/lite/testdata/multi_add.bin";
    //private static final ByteBuffer MULTIPLE_INPUTS_MODEL_BUFFER = HARClassifier.loadModelFile(MULTIPLE_INPUTS_MODEL_PATH);
    public HARClassifier(final Context context) {
            try {
                tflite = new Interpreter(loadModelFile(context));
                //Interpreter interpreter = new Interpreter(MULTIPLE_INPUTS_MODEL_BUFFER);
            }catch (Exception ex){
                ex.printStackTrace();
            }
        }
    private MappedByteBuffer loadModelFile(final Context context) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd("converted_pred_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    public void predictProbabilities(Object[] inputs) {
        outputs.put(0, parsedOutput0);
        outputs.put(1, parsedOutput1);
        tflite.runForMultipleInputsOutputs(inputs, outputs);

        //Log.d("this is my array", "arr: " + Arrays.toString(result));
        //Log.d("Input", "arr: " + Arrays.toString(data));
        //Walking       Upstairs     Downstairs 	Standing_Standing	Running
    }

    public float[][] getQuaternion() {
        return parsedOutput0;
    }

    public float[][] getDisplacment() {
        return parsedOutput1;
    }
}