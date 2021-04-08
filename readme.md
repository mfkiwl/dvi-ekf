# DVI-EKF
Implementation of a loosely-coupled VI-SLAM.


## Offline trajectories
```
python3 plot_traj.py
```
![](img/offline_trajs.PNG)

## Noisy IMU data
```
python3 plot_imu.py
```

To improve: replace hard coded stdev values, add bias, incorporate effects of gravity.

![](img/offline_noisyimu.PNG)

## Kalman filter
### Propagation only
Using initial pose from monocular trajectory and propagating using IMU values.

![](img/traj_only_prop.PNG)
