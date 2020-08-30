package ahmad.abdelqader.har_android;

import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CSVFile {
    InputStream inputStream;
    float[][] tables;
    public CSVFile(InputStream inputStream){
        this.inputStream = inputStream;
        tables = new float[657][128];
    }

    public List read(){
        List resultList = new ArrayList();
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
        int c = 0;
        try {
            String csvLine;
            while ((csvLine = reader.readLine()) != null) {
                String[] row = csvLine.split(",");
                resultList.add(row);
                for(int i = 0; i < 128; i++){
                    tables[c][i] = Float.parseFloat(row[i]);

                }
                c++;
            }
        }
        catch (IOException ex) {
            throw new RuntimeException("Error in reading CSV file: "+ex);
        }
        finally {
            try {
                inputStream.close();
            }
            catch (IOException e) {
                throw new RuntimeException("Error while closing input stream: "+e);
            }
        }

        Log.d("Tabels: ", "arr: " + Arrays.toString(tables));
        return resultList;
    }
    public float[][] getTable()
    {
        return tables;
    }
    public float[] readLabels(){
        float [] labels = new float[657];
        List resultList = new ArrayList();
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
        int c = 0;
        try {
            String csvLine;
            while ((csvLine = reader.readLine()) != null) {
                String[] row = csvLine.split(",");
                resultList.add(row);
                    labels[c] = Float.parseFloat(row[0]);
                c++;
            }
        }
        catch (IOException ex) {
            throw new RuntimeException("Error in reading CSV file: "+ex);
        }
        finally {
            try {
                inputStream.close();
            }
            catch (IOException e) {
                throw new RuntimeException("Error while closing input stream: "+e);
            }
        }

        Log.d("Tabels: ", "arr: " + Arrays.toString(tables));
        return labels;
    }

}
