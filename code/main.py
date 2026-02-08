import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as plt
from load_data import imud, camd, vicd
import torch
import matplotlib as mpl
from mpl_toolkits.mplot3d import art3d
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",  
    
    "font.size": 10,             
    "axes.titlesize": 10,       
    "axes.labelsize": 9,      
    "legend.fontsize": 8,       
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
})
def align_closest(t_imu, t_cam):
    idx = np.searchsorted(t_imu, t_cam)

    idx = np.clip(idx, 1, len(t_imu)-1)

    prev = t_imu[idx-1]
    next = t_imu[idx]

    closer = np.abs(t_cam - prev) <= np.abs(next - t_cam)
    idx[closer] -= 1
    return idx
canvas_h, canvas_w = 540, 1080  
panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

def update_panorama(R, k, panorama):
    img = camd['cam'][:,:,:,k]
    img_ds = img[::, ::, :] # Downsample for speed
    H, W = img_ds.shape[:2] # width, height
    T_SC = np.array([
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0.1],
        [0, 0, 0, 1]
    ])
    T = np.eye(4)  # Create a 4x4 identity matrix
    T[:3, :3] = R  # Assign the rotation matrix to the top-left 3x3 block
    fov_w, fov_h = 60 * np.pi/180, 45 * np.pi/180
    m = np.linspace(0, W-1, W)
    i = np.linspace(0, H-1, H)
    mm, ii = np.meshgrid(m, i)
    # number of pixels to camera frame
    theta_loc   = (mm / (W-1)-0.5) * fov_w  #map num of pixels to fov in radians range from [-fov/2, fov/2]
    phi_loc = (ii / (H-1)-0.5 ) * fov_h  #map num of pixels to fov in radians range from [- fov/2, fov/2]
    X_l = np.sin(theta_loc) * np.cos(phi_loc)
    Y_l = np.sin(phi_loc)
    Z_l = np.cos(theta_loc) * np.cos(phi_loc)
    T = T @ T_SC  
    # Rotate to World Frame (Direction vectors only for panorama)
    xyz_world = T @ np.stack([X_l.ravel(), Y_l.ravel(), Z_l.ravel(), np.ones_like(X_l).ravel()], axis=0)
    xyz_world = xyz_world / np.linalg.norm(xyz_world[:3, :], axis=0, keepdims=True)
    # World vectors to Equirectangular (lon, lat)

    lon = - np.arctan2(xyz_world[1, :], xyz_world[0, :])    # [-pi, pi]
    lat = np.arctan2(xyz_world[2, :], np.sqrt(xyz_world[0, :]**2 + xyz_world[1, :]**2))  # [-pi/2, pi/2]
    canvas_h, canvas_w = panorama.shape[:2]
    px_x = ((lon + np.pi) / (2 * np.pi) * (canvas_w - 1)).astype(int)
    px_y = ((0.5 * np.pi - lat) / np.pi * (canvas_h - 1)).astype(int)
    
    # Flatten colors and update canvas
    colors = img_ds.reshape(-1, 3)
    panorama[px_y, px_x, :] = colors
    
    return panorama
def qconj(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)
def qnormalize(q, eps=1e-12):
    return q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True)+eps)
def qinv(q, eps=1e-12):
    denom = (q * q).sum(dim=-1, keepdim=True)+eps
    return qconj(q) / denom
def qmul(q, r):
    # q, r: (...,4) wxyz
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

def qexp(v, eps=1e-12):
    s = v[..., 0:1]          # (...,1)
    u = v[..., 1:]           # (...,3)

    u_norm = torch.linalg.norm(u, dim=-1, keepdim=True)  # (...,1)
    exp_s  = torch.exp(s)                                 # (...,1)

    # sin(x)/x with safe handling near 0
    sin_over_x = torch.where(
        u_norm > eps,
        torch.sin(u_norm) / u_norm,
        torch.ones_like(u_norm)
    )

    qw = exp_s * torch.cos(u_norm)          # (...,1)
    qv = exp_s * sin_over_x * u             # (...,3)

    return torch.cat([qw, qv], dim=-1)      # (...,4)
def qlog_unit(q, eps=1e-8):
    # q: (...,4) unit quaternion [w,x,y,z]
    w = q[..., 0].clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    v = q[..., 1:]
    v_norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)

    # angle/2 = atan2(||v||, w)  (比 acos(w) 更穩)
    half_theta = torch.atan2(v_norm, w.unsqueeze(-1))  # (...,1)

    return torch.cat([torch.zeros_like(half_theta), v / v_norm * half_theta], dim=-1)
sensitivity_acc = 330.0  # mV/g for accelerometer
sensitivity_gyr = 3.3*180/np.pi # mV/g for gyroscope
Vref = 3300  # reference voltage
scalefactor_acc = Vref/1023/sensitivity_acc
scalefactor_gyr = Vref/1023/sensitivity_gyr
bias_acc_x = np.mean(imud[1,0:100])  # 假設前100個樣本是靜止的
bias_acc_y = np.mean(imud[2,0:100])  # 假設前100個樣本是靜止的
bias_acc_z = np.mean(imud[3,0:100])-1/scalefactor_acc  # 假設前100個樣本是靜止的
bias_gyr_x = np.mean(imud[4,0:100])  # 假設前100個樣本是靜止的
bias_gyr_y = np.mean(imud[5,0:100])  # 假設前100個樣本是靜止的
bias_gyr_z = np.mean(imud[6,0:100])  # 假設前100個樣本是靜止的
tstamp = imud[0, :]-imud[0, 0]
camdat = camd['cam']
tstamp_cam = camd['ts'][0, :]-imud[0, 0]
Ax = (imud[1, :]- bias_acc_x)*scalefactor_acc
Ay = (imud[2, :]- bias_acc_y)*scalefactor_acc
Az = (imud[3, :]- bias_acc_z)*scalefactor_acc
Wx = (imud[4, :]- bias_gyr_x)*scalefactor_gyr
Wy = (imud[5, :]- bias_gyr_y)*scalefactor_gyr
Wz = (imud[6, :]- bias_gyr_z)*scalefactor_gyr
acc_t = torch.from_numpy(np.stack([Ax, Ay, Az], axis=-1)).to(dtype=torch.float64)
Wx_t = torch.from_numpy(np.asarray(Wx)).to(dtype=torch.float64)
Wy_t = torch.from_numpy(np.asarray(Wy)).to(dtype=torch.float64)
Wz_t = torch.from_numpy(np.asarray(Wz)).to(dtype=torch.float64)
t_t  = torch.from_numpy(np.asarray(tstamp)).to(dtype=torch.float64)
Vstamp = vicd['ts'][0,:]-imud[0,0]
V = vicd['rots']
pitch = np.zeros(V.shape[2])
roll = np.zeros(V.shape[2])
yaw = np.zeros(V.shape[2])
pitch_test = np.zeros(V.shape[2])
roll_test = np.zeros(V.shape[2])
yaw_test = np.zeros(V.shape[2])

for i in range(V.shape[2]):
    pitch[i] = -np.arcsin(V[2,0,i])

    # gimbal lock
    if abs(abs(V[2,0,i]) - 1.0) < 1e-6:
        # pitch ~ +/- 90deg，yaw/roll
        yaw[i] = 0.0
        if V[2,0,i] < 0:  # pitch = +pi/2
            roll[i] = np.arctan2(-V[0,1,i], V[0,2,i])
        else:             # pitch = -pi/2
            roll[i] = np.arctan2(V[0,1,i], -V[0,2,i])
    else:
        # roll, yaw 用 atan2（保象限、穩定）
        roll[i] = np.arctan2(V[2,1,i], V[2,2,i])
        yaw[i]  = np.arctan2(V[1,0,i], V[0,0,i])
    # pitch_test[i] = t3d.euler.mat2euler(V[:,:,i], axes='rzyx')[1] 
    # roll_test[i] = t3d.euler.mat2euler(V[:,:,i], axes='rzyx')[2]
    # yaw_test[i] = t3d.euler.mat2euler(V[:,:,i], axes='rzyx')[0]
yaw = np.unwrap(yaw)
pitch = np.unwrap(pitch)
roll = np.unwrap(roll)

N = len(tstamp)
qt = torch.zeros((N, 4), dtype=torch.float64)          # (N,4) wxyz
qt[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
roll_imu = np.zeros(len(tstamp))
pitch_imu = np.zeros(len(tstamp))
yaw_imu = np.zeros(len(tstamp))
for m in range(N-1):
    dt = tstamp[m+1]-tstamp[m]
    # qt[m] = np.array([qs_t1, qv_t1[0], qv_t1[1], qv_t1[2]])
    omega = torch.stack([torch.tensor(0, dtype=Wx_t.dtype, device=Wx_t.device), Wx_t[m], Wy_t[m], Wz_t[m]])        # (3,)
    v = 0.5 * omega * dt                               # (3,)  = omega*dt/2
    dq = qexp(v)                                  # (4,)
    qt[m+1] = qmul(qt[m], dq)                           # (4,)
    q_np = qt[m+1].detach().cpu().numpy()
    y, p, r = t3d.euler.quat2euler(q_np, axes='rzyx')
    yaw_imu[m], pitch_imu[m], roll_imu[m] = y, p, r
    # print(yaw_imu[m], pitch_imu[m], roll_imu[m])
yaw_imu = np.unwrap(yaw_imu)
pitch_imu = np.unwrap(pitch_imu)
roll_imu = np.unwrap(roll_imu)
firsterm = 0
secterm = 0
def cost_func(qt):
    N = qt.shape[0]

    dt = (t_t[1:] - t_t[:-1]).unsqueeze(-1)                           # (N-1,1)
    omega = torch.stack([torch.zeros_like(Wx_t[:-1]), Wx_t[:-1], Wy_t[:-1], Wz_t[:-1]], dim=-1)  # (N-1, 4)  
    v = 0.5 * omega * dt                               # (3,)  = omega*dt/2
    dq = qexp(v)                                                 # (N-1,4)
    # print(dq.shape)
    # print(qt[:-1].shape)
    fqt = qmul(qt[:-1], dq)                                           # (N-1,4)
    x = qmul(qinv(qt[1:]), fqt)                                       # (N-1,4)
    firsterm = torch.linalg.vector_norm(2*qlog_unit(x), dim=-1).pow(2).sum()

    g = torch.tensor([0,0,0,1], dtype=qt.dtype, device=qt.device).expand(N-1,4)
    hq = qmul(qmul(qinv(qt[1:]), g), qt[1:])                           # (N-1,4)
    a  = torch.cat([torch.zeros((N-1,1), dtype=qt.dtype, device=qt.device),
                    acc_t[1:]], dim=-1)                                # (N-1,4)

    secterm = torch.linalg.vector_norm(a - hq, dim=-1).pow(2).sum()    
    if not torch.isfinite(firsterm) or not torch.isfinite(secterm):
        print("firsterm finite?", torch.isfinite(firsterm).item(),
            "secterm finite?", torch.isfinite(secterm).item())
    return 0.5*firsterm + 0.5*secterm
alpha = 1e-2
for it in range(1000):
    qt = qt.detach().clone().requires_grad_(True)

    loss_before = cost_func(qt)
    grad_qt, = torch.autograd.grad(loss_before, qt)

    with torch.no_grad():
        qt = qt - alpha * grad_qt                                      
        qt = qt / torch.linalg.vector_norm(qt, dim=-1, keepdim=True).clamp_min(1e-12)
        qt[0] = torch.tensor([1.0,0.0,0.0,0.0], dtype=qt.dtype, device=qt.device)
        loss_after = cost_func(qt)
    print(f"iter {it}: loss_before={loss_before.item():.6f}, loss_after={loss_after.item():.6f}, grad_absmax={grad_qt.abs().max().item():.3e}")

    if abs(loss_after - loss_before)/loss_before < 1e-6:
        print(f"Converged at iteration {it}")
        break
q_np = qt.detach().cpu().numpy()

yaw_opt = np.zeros(len(tstamp))
pitch_opt = np.zeros(len(tstamp))
roll_opt = np.zeros(len(tstamp))
for m in range(len(tstamp)):    
    yaw_opt[m] = t3d.euler.quat2euler(q_np[m], axes='rzyx')[0]
    pitch_opt[m] = t3d.euler.quat2euler(q_np[m], axes='rzyx')[1]
    roll_opt[m] = t3d.euler.quat2euler(q_np[m], axes='rzyx')[2]
yaw_opt = np.unwrap(yaw_opt)
pitch_opt = np.unwrap(pitch_opt)
roll_opt = np.unwrap(roll_opt)

yaw_diff_opt = yaw_opt - yaw[:len(yaw_opt)]
yaw_diff = yaw_imu - yaw[:len(yaw_imu)]
rmse_raw = np.sqrt(np.mean(yaw_diff**2))
rmse_opt = np.sqrt(np.mean(yaw_diff_opt**2))
improvement_yaw = (rmse_raw - rmse_opt) / rmse_raw * 100
roll_diff_opt = roll_opt - roll[:len(roll_opt)]
roll_diff_raw = roll_imu - roll[:len(roll_imu)]

# Calculate RMSE
rmse_roll_raw = np.sqrt(np.mean(roll_diff_raw**2))
rmse_roll_opt = np.sqrt(np.mean(roll_diff_opt**2))

# Calculate Improvement %
improvement_roll = (rmse_roll_raw - rmse_roll_opt) / rmse_roll_raw * 100


# --- 2. PITCH Calculation ---
# Calculate differences (ensure lengths match)
pitch_diff_opt = pitch_opt - pitch[:len(pitch_opt)]
pitch_diff_raw = pitch_imu - pitch[:len(pitch_imu)]

# Calculate RMSE
rmse_pitch_raw = np.sqrt(np.mean(pitch_diff_raw**2))
rmse_pitch_opt = np.sqrt(np.mean(pitch_diff_opt**2))

# Calculate Improvement %
improvement_pitch = (rmse_pitch_raw - rmse_pitch_opt) / rmse_pitch_raw * 100


# --- 3. Print Results ---
print(f"--- Roll Improvement ---")
print(f"Raw RMSE: {rmse_roll_raw:.4f} rad")
print(f"Opt RMSE: {rmse_roll_opt:.4f} rad")
print(f"Improvement: {improvement_roll:.2f}%\n")

print(f"--- Pitch Improvement ---")
print(f"Raw RMSE: {rmse_pitch_raw:.4f} rad")
print(f"Opt RMSE: {rmse_pitch_opt:.4f} rad")
print(f"Improvement: {improvement_pitch:.2f}%\n")

print(f"--- Yaw Improvement (from your code) ---")
print(f"Improvement: {improvement_yaw:.2f}%")

## plotting



plt.figure(figsize=(3.5, 3))
plt.subplot(3, 1, 1)
plt.plot(Vstamp, roll*180/np.pi, label='True roll(Blue)', linewidth=1, color='blue')
plt.plot(tstamp, roll_opt*180/np.pi, label='Estimated roll(Red)', linewidth=1, color='red')
# plt.plot(tstamp, roll_imu*180/np.pi, label='Estimated roll(Red)', linewidth=1, color='red')
plt.title('True roll(Blue) vs Estimated roll(Red)')
plt.ylabel('Angle (deg)')
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(Vstamp, pitch*180/np.pi, label='True Pitch(Blue)', linewidth=1, color='blue')
plt.plot(tstamp, pitch_opt*180/np.pi, label='Estimated Pitch(Red)', linewidth=1, color='red')
# plt.plot(tstamp, pitch_imu*180/np.pi, label='Estimated Pitch(Red)', linewidth=1, color='red')
plt.ylabel('Angle (deg)')
plt.title('True Pitch(Blue) vs Estimated Pitch(Red)')
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(Vstamp, yaw*180/np.pi, label='True Yaw(Blue)', linewidth=1, color='blue')
plt.plot(tstamp, yaw_opt*180/np.pi, label='Estimated Yaw(Red)', linewidth=1, color='red')
# plt.plot(tstamp, yaw_imu*180/np.pi, label='Estimated Yaw(Red)', linewidth=1, color='red')
plt.title('True Yaw(Blue) vs Estimated Yaw(Red)')
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
# plt.tight_layout()
plt.subplots_adjust(
    left=0.15,
    right=0.95,
    bottom=0.15,
    top=0.9,
    hspace=1.0
)
# plt.savefig("after optimized dataset 11.pdf", bbox_inches="tight")
# plt.show(block=True)


imu_idx = align_closest(tstamp, tstamp_cam)
vicon_idx = align_closest(Vstamp, tstamp_cam)
t_sc = np.array([0, 0, 0.1])
canvas_h, canvas_w = 540, 1080 
panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
t_sc = np.array([0, 0, 0.1])

print("Starting Panorama Stitching...")
for k in range(0,len(tstamp_cam), 10): # Skip frames to speed up if needed
    idx = imu_idx[k]
    R_ws = t3d.quaternions.quat2mat(q_np[idx])  # World to Sensor
    R_vicon = V[:,:,vicon_idx[k]]
    # panorama = update_panorama(R_ws, k, panorama)
    panorama = update_panorama(R_vicon, k, panorama)
    if k % 100 == 0:
        print(f"Processed frame {k+1}/{len(tstamp_cam)}")

# Final Plotting with Radian Axes
plt.figure(figsize=(7,3))

# extent sets the axis to radians: [left, right, bottom, top]
plt.imshow(panorama, extent=[-np.pi, np.pi, -np.pi/2, np.pi/2])

plt.title("2D Equirectangular Panorama (World Frame)")
plt.xlabel("Longitude λ (rad)")
plt.ylabel("Latitude φ (rad)")

# Set Radian Ticks
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
           [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.yticks([-np.pi/2, 0, np.pi/2], 
           [r'$-\pi/2$', '0', r'$\pi/2$'])
plt.subplots_adjust(
    left=0,
    right=1,
    bottom=0.175,
    top=0.9,
    hspace=1.0
)
plt.grid(True, color='white', linestyle='--', alpha=0.3)
# plt.savefig("ground truth 9.pdf", bbox_inches="tight")
# plt.savefig("estimated paronama 9.pdf", bbox_inches="tight")
plt.show(block=True)

# plt.subplot(2, 1, 2)
# plt.plot(Vstamp, pitch, label='pitch')
# plt.plot(Vstamp, roll, label='roll')
# plt.plot(Vstamp, yaw, label='yaw')
# # plt.plot(tstamp, pitch_imu, '--', label='pitch imu')
# # plt.plot(tstamp, roll_imu, '--', label='roll imu')
# # plt.plot(tstamp, yaw_imu, '--', label='yaw imu')
# plt.plot(tstamp, pitch_opt, '--', label='pitch imu')
# plt.plot(tstamp, roll_opt, '--', label='roll imu')
# plt.plot(tstamp, yaw_opt, '--', label='yaw imu')

# # plt.plot(Vstamp, pitch_test, ':', label='pitch test from mat2euler')
# # plt.plot(Vstamp, roll_test, ':', label='roll test from mat2euler')
# # plt.plot(Vstamp, yaw_test, ':', label='yaw test from mat2euler')
# plt.subplots_adjust(
#     left=0.15,
#     right=0.95,
#     bottom=0.15,
#     top=0.9,
#     hspace=1.0
# )

# plt.figure(figsize=(3.5, 3))
# plt.subplot(3, 1, 1)
# plt.plot(Vstamp, roll*180/np.pi, label='True roll(Blue)', linewidth=1, color='blue')
# # plt.plot(tstamp, roll_opt*180/np.pi, label='Estimated roll(Red)', linewidth=1, color='red')
# plt.plot(tstamp, roll_imu*180/np.pi, label='Estimated roll(Red)', linewidth=1, color='red')
# plt.title('True roll(Blue) vs Estimated roll(Red)')
# plt.ylabel('Angle (deg)')
# plt.grid(True)
# plt.subplot(3, 1, 2)
# plt.plot(Vstamp, pitch*180/np.pi, label='True Pitch(Blue)', linewidth=1, color='blue')
# # plt.plot(tstamp, pitch_opt*180/np.pi, label='Estimated Pitch(Red)', linewidth=1, color='red')
# plt.plot(tstamp, pitch_imu*180/np.pi, label='Estimated Pitch(Red)', linewidth=1, color='red')
# plt.ylabel('Angle (deg)')
# plt.title('True Pitch(Blue) vs Estimated Pitch(Red)')
# plt.grid(True)
# plt.subplot(3, 1, 3)
# plt.plot(Vstamp, yaw*180/np.pi, label='True Yaw(Blue)', linewidth=1, color='blue')
# # plt.plot(tstamp, yaw_opt*180/np.pi, label='Estimated Yaw(Red)', linewidth=1, color='red')
# plt.plot(tstamp, yaw_imu*180/np.pi, label='Estimated Yaw(Red)', linewidth=1, color='red')
# plt.title('True Yaw(Blue) vs Estimated Yaw(Red)')
# plt.grid(True)
# plt.xlabel('Time (s)')
# plt.ylabel('Angle (deg)')
# # plt.tight_layout()
# plt.subplots_adjust(
#     left=0.15,
#     right=0.95,
#     bottom=0.15,
#     top=0.9,
#     hspace=1.0
# )
# # plt.savefig("before optimized dataset 10.pdf", bbox_inches="tight")
# plt.show(block=True)

# plt.figure(figsize=(3.5, 3))
# # plt.subplot(2, 1, 1)
# # plt.plot(tstamp, acc_t[:,0], label='Ax')
# # plt.plot(tstamp, acc_t[:,1], label='Ay')
# # plt.plot(tstamp, acc_t[:,2], label='Az')

# plt.plot(tstamp, Wx_t, label='roll rate Wx')
# plt.plot(tstamp, Wy_t, label='pitch rate Wy')
# plt.plot(tstamp, Wz_t, label='yaw rate Wz')
# plt.grid(True)
# plt.legend()
# plt.title('IMU Data')
# plt.xlabel('Time (s)')
# plt.ylabel('Angular Rate (rad/s)')