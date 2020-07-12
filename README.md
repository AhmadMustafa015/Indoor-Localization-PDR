# Indoor-Localization-PDR
## This repository contains an android navigation App based on Pedestrian Dead Reckoning (PDR) algorithms. 

how it works?
first calculate the bias for gyro and magnetometer:
1- gyro: put your phone on a flate surface for number of trials = 600: 
at first trials the gyroBias = raw gyro
for the other trials the bias is calculated from taking the moving avg for all the gyro data.
at the end we will have a gyroBias array with 3 float data. // TODO: calulate better bias
2- magnetometer: the objective here is to calculate the hard iron error [vx,vy,vz] and the magnatic field stregth the equation which have all those parameter is:
B = [B0,B1,B2,B3] = [2vx,2vy,2vz,B2-Vx2-vy2-vz2]
locus of magnetometer measurements represents the primary information available to the
calibration algorithms to determine the hard- and soft-iron calibration V and W-1. Under arbitrary rotation
of the smartphone by its owner. //TODO: implement soft iron equation 10 parameter to find
From Pg. 14 of "Calibrating an eCompass in the Presence of Hard and Soft-iron Interference, Rev. 3" we need to calculate 
β (XTX) –1= XTY β is the solution vector from which the four calibration parameters can be easily determined
if first run  reserveX = rawMagneticValues[0];
            reserveY = rawMagneticValues[1];
            reserveZ = rawMagneticValues[2];
else 	x = reserveX;
            y = reserveY;
            z = reserveZ;

            reserveX = rawMagneticValues[0];
            reserveY = rawMagneticValues[1];
            reserveZ = rawMagneticValues[2];
XTX = new double[][]{{XTX[0][0] + x * x, XTX[0][1] + x * y, XTX[0][2] + x * z, XTX[0][3] + x},
                             {XTX[1][0] + x * y, XTX[1][1] + y * y, XTX[1][2] + y * z, XTX[1][3] + y},
                             {XTX[2][0] + x * z, XTX[2][1] + y * z, XTX[2][2] + z * z, XTX[2][3] + z},
                             {XTX[3][0] + x,     XTX[3][1] + y,     XTX[3][2] + z,     XTX[3][3] + 1}};

        XTY = new double[][] {{XTY[0][0] + x * (x * x + y * y + z * z)},
                              {XTY[1][0] + y * (x * x + y * y + z * z)},
                              {XTY[2][0] + z * (x * x + y * y + z * z)},
                              {XTY[3][0] + (x * x + y * y + z * z)}};
After we have XTX and XTY we can calculate B
at the end we will return xBias, yBias,zBias, magFieldStrength (array of 4 float)

find stepCounter sensitivity?
Now when we click the button:
first get intial heading with repect to earth using magnetometer data
float[][] initialOrientation = MagneticFieldOrientation.getOrientationMatrix(currGravity, currMag, magBias);
initialHeading = MagneticFieldOrientation.getHeading(currGravity, currMag, magBias);
inside MagneticFieldOrientation.getOrientationMatrix:
first remove bias by subtract magnX - magnXBias ...
Then convert to matrix {{array[0]},{array[1]},{array[2]}}
This application note uses the industry standard “NED” (North, East, Down) coordinate system to label
axes on the mobile phone and the IMU use ENU a simple conversion by using this matrix
{{0,1,0},
{1,0,0},
{0,0,-1}}
we did the rotation for gravity vector and magnatometer
then calculate the roll and pitch angle from the gravity as specified in the report:
double G_r = Math.atan2(G_m_values[1][0], G_m_values[2][0]);
double G_p = Math.atan2(-G_m_values[0][0], G_m_values[1][0] * Math.sin(G_r) + G_m_values[2][0] * Math.cos(G_r));
With the angles roll and pitch known from the accelerometer, the magnetometer reading can be de-rotated to
correct for the phone orientation using:  R_rp = {{Math.cos(G_p), Math.sin(G_p) * Math.sin(G_r), Math.sin(G_p) * Math.cos(G_r)},
                            {0, Math.cos(G_r), -Math.sin(G_r)},
                            {-Math.sin(G_p), Math.cos(G_p) * Math.sin(G_r), Math.cos(G_p) * Math.cos(G_r)}};
rotationMatrix * biasedMagnatometer = The vector represent the components of the magnetometer sensor after correcting for the Hard-Iron
offset and after de-rotating to the flat plane where θ = φ = 0.

h = -1*(Math.atan2(-m_M_rp.get(1), m_M_rp.get(0)) + 11.0 * Math.PI/180.0); in rad
allows solution for the yaw angle ψ where ψ is computed relative to magnetic north. The yaw
angle ψ is therefore the required tilt-compensated eCompass heading.
we added to this angle a magnetic declination because of the differance in north and true north ex in Turkey 5.72. we multiply the angle by -1 so that if we move 
to the east the angle will be -ve if we move to the west it will be +ve.
After this we calculate the rotation matrix from yaw and then multiply the biased magnetometer with the rotation matrix the result will be return.
Also tan-1 of the orientationMatrix[1],[0] is used to return the heading.
next step save intial oriantation data (init_Gravity,init_Mag,mag_Bias,gyro_Bias,init_Orientation,init_Heading)

