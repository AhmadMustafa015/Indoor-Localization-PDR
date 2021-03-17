package ahmad.abdelqader.navigation_deeplearning.graph;

import android.content.Context;
import android.graphics.Color;

import org.achartengine.ChartFactory;
import org.achartengine.GraphicalView;
import org.achartengine.chart.PointStyle;
import org.achartengine.model.XYMultipleSeriesDataset;
import org.achartengine.model.XYSeries;
import org.achartengine.renderer.XYMultipleSeriesRenderer;
import org.achartengine.renderer.XYSeriesRenderer;

import java.util.ArrayList;

public class ScatterPlot {

    private String seriesName;
    private ArrayList<Double> xListG;
    private ArrayList<Double> yListG;
    /*private ArrayList<Double> xListM;
    private ArrayList<Double> yListM;
    private ArrayList<Double> xListC;
    private ArrayList<Double> yListC;*/

    public ScatterPlot(String seriesName) {
        this.seriesName = seriesName;
        xListG = new ArrayList<>();
        yListG = new ArrayList<>();
        /*xListM = new ArrayList<>();
        yListM = new ArrayList<>();
        xListC = new ArrayList<>();
        yListC = new ArrayList<>();*/
    }

    public GraphicalView getGraphView(Context context) {

        XYSeries mySeriesG;
        // XYSeries mySeriesC;
        // XYSeries mySeriesM;

        XYSeriesRenderer myRendererG;
        //XYSeriesRenderer myRendererM;
        // XYSeriesRenderer myRendererC;
        XYMultipleSeriesDataset myMultiSeries;
        XYMultipleSeriesRenderer myMultiRenderer;

        //adding the x-axis data from an ArrayList to a standard array
        double[] xSetG = new double[xListG.size()];
        for (int i = 0; i < xListG.size(); i++)
            xSetG[i] = xListG.get(i);
        //adding the x-axis data from an ArrayList to a standard array
        /*double[] xSetM = new double[xListM.size()];
        for (int i = 0; i < xListM.size(); i++)
            xSetM[i] = xListM.get(i);
        //adding the x-axis data from an ArrayList to a standard array
        double[] xSetC = new double[xListC.size()];
        for (int i = 0; i < xListC.size(); i++)
            xSetC[i] = xListC.get(i);
        //adding the y-axis data from an ArrayList to a standard array*/
        double[] ySetG = new double[yListG.size()];
        for (int i = 0; i < yListG.size(); i++)
            ySetG[i] = yListG.get(i);
        //adding the y-axis data from an ArrayList to a standard array
       /* double[] ySetM = new double[yListM.size()];
        for (int i = 0; i < yListM.size(); i++)
            ySetM[i] = yListM.get(i);
        //adding the y-axis data from an ArrayList to a standard array
        double[] ySetC = new double[yListC.size()];
        for (int i = 0; i < yListC.size(); i++)
            ySetC[i] = yListC.get(i);*/

        //creating a new sequence using the x-axis and y-axis data
        mySeriesG = new XYSeries("Gyro");
        for (int i = 0; i < xSetG.length; i++)
            mySeriesG.add(xSetG[i], ySetG[i]);

       /* mySeriesM = new XYSeries("Magnetometer");
        for (int i = 0; i < xSetM.length; i++)
            mySeriesM.add(xSetM[i], ySetM[i]);

        mySeriesC = new XYSeries("Complementary");
        for (int i = 0; i < xSetC.length; i++)
            mySeriesC.add(xSetC[i], ySetC[i]);*/

        //defining chart visual properties
        myRendererG = new XYSeriesRenderer();
        myRendererG.setFillPoints(true);
        myRendererG.setPointStyle(PointStyle.CIRCLE);
//        myRendererG.setColor(Color.GREEN);
        myRendererG.setColor(Color.parseColor("#FF9A14FF"));

      /*  myRendererM = new XYSeriesRenderer();
        myRendererM.setFillPoints(true);
        myRendererM.setPointStyle(PointStyle.SQUARE);
        myRendererM.setColor(Color.parseColor("#FFFF1919"));

        myRendererC = new XYSeriesRenderer();
        myRendererC.setFillPoints(true);
        myRendererC.setPointStyle(PointStyle.TRIANGLE);
        myRendererC.setColor(Color.parseColor("#FF9A14FF"));*/

        myMultiSeries = new XYMultipleSeriesDataset();
        myMultiSeries.addSeries(mySeriesG);
        // myMultiSeries.addSeries(mySeriesM);
        // myMultiSeries.addSeries(mySeriesC);
        myMultiRenderer = new XYMultipleSeriesRenderer();
        myMultiRenderer.addSeriesRenderer(myRendererG);
        //  myMultiRenderer.addSeriesRenderer(myRendererM);
        //  myMultiRenderer.addSeriesRenderer(myRendererC);
        //setting text graph element sizes
        myMultiRenderer.setPointSize(10); //size of scatter plot points
        myMultiRenderer.setShowLegend(false); //show legend
        myMultiRenderer.setLegendTextSize(40);
        myMultiRenderer.setLegendHeight(100);
        //set chart and label sizes
        //myMultiRenderer.setChartTitle("Position");
        //myMultiRenderer.setChartTitleTextSize(75);
        myMultiRenderer.setLabelsTextSize(40);

        //setting X labels and Y labels position
        int[] chartMargins = {100, 100, 25, 100}; //top, left, bottom, right
        myMultiRenderer.setMargins(chartMargins);
        myMultiRenderer.setYLabelsPadding(50);
        myMultiRenderer.setXLabelsPadding(10);

        //setting chart min/max
        double bound = getMaxBound();
        myMultiRenderer.setXAxisMin(-bound);
        myMultiRenderer.setXAxisMax(bound);
        myMultiRenderer.setYAxisMin(-bound);
        myMultiRenderer.setYAxisMax(bound);
        //myMultiRenderer.setBackgroundColor(Color.WHITE);
        //myMultiRenderer.setApplyBackgroundColor(true);
        myMultiRenderer.setAxesColor(Color.RED);
        //returns the graphical view containing the graphz
        return ChartFactory.getScatterChartView(context, myMultiSeries, myMultiRenderer);
    }

    //add a point to the series
    public void addPoint(double x, double y) {
        xListG.add(x);
        yListG.add(y);
    }
    /* public void addPointM(double x, double y) {
         xListM.add(x);
         yListM.add(y);
     }
     public void addPointC(double x, double y) {
         xListC.add(x);
         yListC.add(y);
     }*/
    public float getLastXPoint() {
        double x = xListG.get(xListG.size() - 1);
        return (float)x;
    }
    /*public float getLastXPointM() {
        double x = xListM.get(xListM.size() - 1);
        return (float)x;
    }
    public float getLastXPointC() {
        double x = xListC.get(xListC.size() - 1);
        return (float)x;
    }*/
    public float getLastYPoint() {
        double y = yListG.get(yListG.size() - 1);
        return (float)y;
    }
    /*public float getLastYPointM() {
        double y = yListM.get(yListM.size() - 1);
        return (float)y;
    }
    public float getLastYPointC() {
        double y = yListC.get(yListC.size() - 1);
        return (float)y;
    }*/
    public void clearSet() {
        xListG.clear();
        yListG.clear();
       /* xListM.clear();
        yListM.clear();
        xListC.clear();
        yListC.clear();*/
    }

    private double getMaxBound() {
        double max = 0;
        for (double num : xListG)
            if (max < Math.abs(num))
                max = num;
        for (double num : yListG)
            if (max < Math.abs(num))
                max = num;
        return (Math.abs(max) / 100) * 100 + 10; //rounding up to the nearest hundred
    }
}
