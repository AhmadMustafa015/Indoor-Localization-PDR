package edu.onbasli.indoorlocalization.InertiaNavegation.prefs;

import android.content.Context;
import android.content.SharedPreferences;
import android.hardware.SensorManager;
import android.preference.PreferenceManager;

import edu.onbasli.indoorlocalization.InertiaNavegation.activity.config.FilterConfigActivity;


public class PrefUtils
{
	public static boolean getPrefKalmanFilterAccEnabled(Context context) {
		SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
		return prefs.getBoolean(FilterConfigActivity.ANDROID_LINEAR_ACCEL_ENABLED_KEY, true);
	}
	public static boolean getPrefGPSPowerSaving(Context context) {
		SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
		return prefs.getBoolean(FilterConfigActivity.GPS_POWER_SAVING_MODE, false);
	}

    public static float getPrefFSensorLpfLinearAccelerationTimeConstant(Context context) {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
        return Float.parseFloat(prefs.getString(FilterConfigActivity.FSENSOR_LPF_LINEAR_ACCEL_TIME_CONSTANT_KEY, String.valueOf(0.5f)));
    }

    public static boolean getPrefFSensorComplimentaryLinearAccelerationEnabled(Context context) {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
        return prefs.getBoolean(FilterConfigActivity.FSENSOR_COMPLIMENTARY_LINEAR_ACCEL_ENABLED_KEY, false);
    }

    public static float getPrefFSensorComplimentaryLinearAccelerationTimeConstant(Context context) {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
        return Float.parseFloat(prefs.getString(FilterConfigActivity.FSENSOR_COMPLIMENTARY_LINEAR_ACCEL_TIME_CONSTANT_KEY, String.valueOf(0.5f)));
    }

    public static boolean getPrefFSensorKalmanLinearAccelerationEnabled(Context context) {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
        return prefs.getBoolean(FilterConfigActivity.FSENSOR_KALMAN_LINEAR_ACCEL_ENABLED_KEY, true);
    }

    public static boolean getPrefLpfSmoothingEnabled(Context context) {
		SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
		return prefs.getBoolean(FilterConfigActivity.LPF_SMOOTHING_ENABLED_KEY, true);
	}

    public static float getPrefLpfSmoothingTimeConstant(Context context) {
		SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
		return Float.parseFloat(prefs.getString(FilterConfigActivity.LPF_SMOOTHING_TIME_CONSTANT_KEY, String.valueOf(0.5f)));
	}

    public static boolean getPrefMeanFilterSmoothingEnabledAcc(Context context) {
		SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
		return prefs.getBoolean(FilterConfigActivity.MEAN_FILTER_SMOOTHING_ENABLED_KEY, false);
	}
	public static boolean getPrefMeanFilterSmoothingEnabled(Context context) {
		SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
		return prefs.getBoolean(FilterConfigActivity.MEAN_FILTER_SMOOTHING_ENABLED_KEY_ROTATION, false);
	}

    public static float getPrefMeanFilterSmoothingTimeConstant(Context context) {
		SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
		return Float.parseFloat(prefs.getString(FilterConfigActivity.MEAN_FILTER_SMOOTHING_TIME_CONSTANT_KEY_ROTATION, String.valueOf(0.5f)));
	}
	public static float getPrefMeanFilterSmoothingTimeConstantAcc(Context context) {
		SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
		return Float.parseFloat(prefs.getString(FilterConfigActivity.MEAN_FILTER_SMOOTHING_TIME_CONSTANT_KEY, String.valueOf(0.5f)));
	}

    public static boolean getPrefMedianFilterSmoothingEnabled(Context context) {
		SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
		return prefs.getBoolean(FilterConfigActivity.MEDIAN_FILTER_SMOOTHING_ENABLED_KEY, false);
	}

    public static float getPrefMedianFilterSmoothingTimeConstant(Context context) {
		SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
		return Float.parseFloat(prefs.getString(FilterConfigActivity.MEDIAN_FILTER_SMOOTHING_TIME_CONSTANT_KEY, String.valueOf(0.5f)));
	}

    public static int getSensorFrequencyPrefs(Context context) {
		SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
		return Integer.parseInt(prefs.getString(FilterConfigActivity.SENSOR_FREQUENCY_KEY, String.valueOf(SensorManager.SENSOR_DELAY_FASTEST)));
	}
}
