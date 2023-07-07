import numpy as np
import matplotlib.pyplot as plt

nx = 31
x = np.linspace(0,3,nx)

dx = x[2] - x[1]
gamma = 1.4
gamma_inverse = 1/gamma;
nt = 1400
cfl = 0.5
total_time = 0;

#initial profile
rho = 1 - (0.3146*x)
T =   1 - (0.2314*x)
v = (0.1 + 1.09*x)*(T**0.5)
A = 1 + 2.2*((x-1.5)**2)
print(v)
print(T)
print(rho)

dv_dt_p = np.zeros((nx))
drho_dt_p = np.zeros((nx))
dT_dt_p = np.zeros((nx))

dv_dt_c = np.zeros((nx))
drho_dt_c = np.zeros((nx))
dT_dt_c = np.zeros((nx))

mass_flow_rate_initial = rho*v*A
throat = np.where(A==1)
throat = throat[0]
rho[throat]

mass_flow = np.array((nx))
pressure = np.array((nx))
mach_number = np.array((nx))

rho_old = np.array((nx))
v_old = np.array((nx))
T_old = np.array((nx))

rho_old = rho[:]
v_old = v[:]
T_old = T[:]
T_old

rho_throat = np.zeros((nt))
pressure_throat = np.zeros((nt))
velocity_throat = np.zeros((nt))
temp_throat = np.zeros((nt))
mass_flow_throat = np.zeros((nt))
mach_number_throat = np.zeros((nt))

dv_dt_throat = np.zeros((nt))
drho_dt_throat = np.zeros((nt))

mass_flow_50 = np.zeros((nx))
mass_flow_100 = np.zeros((nx))
mass_flow_150 = np.zeros((nx))
mass_flow_200 = np.zeros((nx))
mass_flow_700 = np.zeros((nx))

for time in range(nt):

    rho_old = rho.copy()
    T_old = T.copy()
    v_old = v.copy()

    dt=np.min(0.5*dx/(T[1:]**0.5+v[1:]))
    
    #print(f"dt = {dt}")
    
    dv_dt_av = np.zeros((nx))
    drho_dt_av = np.zeros((nx))
    dT_dt_av = np.zeros((nx))

    #Running the predictor loop
    for i in range(1,nx-1):
        dv_dx = (v[i+1] - v[i])/dx                     #dv_dx[1:] = v[2:-1] - v[1:-2]
        dA_dx = (np.log(A[i+1]) - np.log(A[i]))/dx      
        drho_dx = (rho[i+1]-rho[i])/dx
        dT_dx = (T[i+1] - T[i])/dx

        t_rho = T[i]/rho[i]
    
        drho_dt_p[i] = -(rho[i]*dv_dx) -(rho[i]*v[i]*dA_dx) - (v[i]*drho_dx)
        dv_dt_p[i] = -(v[i]*dv_dx) - (gamma_inverse*(dT_dx + t_rho*drho_dx))
        dT_dt_p[i] = -(v[i]*dT_dx) - ((gamma - 1)*T[i]*(dv_dx + (v[i]*dA_dx)))
    
        #solution update    
        v[i] = v[i] + dv_dt_p[i]*dt
        rho[i] = rho[i] + drho_dt_p[i]*dt
        T[i] = T[i] + dT_dt_p[i]*dt 
    
    #Corrector step
    for i in range(nx-2,0,-1):
        dv_dx = (v[i] - v[i-1])/dx
        dA_dx = (np.log(A[i]) - np.log(A[i-1]))/dx      
        drho_dx = (rho[i]-rho[i-1])/dx
        dT_dx = (T[i] - T[i-1])/dx

        t_rho = T[i]/rho[i]
        
        drho_dt_c[i] = -(rho[i]*dv_dx) - (rho[i]*v[i]*dA_dx) - (v[i]*drho_dx)
        dv_dt_c[i] = -(v[i]*dv_dx) - (gamma_inverse*(dT_dx + t_rho*drho_dx))
        dT_dt_c[i] = -(v[i]*dT_dx) - (gamma - 1)*T[i]*(dv_dx + (v[i]*dA_dx))

    #solution update
        dv_dt_av[i] = 0.5*(dv_dt_p[i] + dv_dt_c[i])
        drho_dt_av[i] = 0.5*(drho_dt_p[i] + drho_dt_c[i])
        dT_dt_av = 0.5*(dT_dt_p[i] + dT_dt_c[i])
        
        v[i] = v_old[i] + 0.5*(dv_dt_p[i] + dv_dt_c[i])*dt
        T[i] = T_old[i] + 0.5*(dT_dt_p[i] + dT_dt_c[i])*dt
        rho[i] = rho_old[i] + 0.5*(drho_dt_p[i] + drho_dt_c[i])*dt
    
    #inlet
    rho[0] = 1
    T[0] = 1
    v[0] = 2*v[1] - v[2]

    #outlet
    rho[nx - 1] = 2*rho[nx-2] - rho[nx-3]
    v[nx-1] = 2*v[nx-2] - v[nx-3]
    T[nx - 1] = 2*T[nx-2] - T[nx-3]

    mass_flow = rho*A*v
    pressure = rho*T
    mach_number = v/T**0.5

    rho_throat[time] = rho[throat]
    velocity_throat[time] = v[throat]
    pressure_throat[time] = pressure[throat]
    temp_throat[time] = T[throat]
    mach_number_throat[time] = mach_number[throat]
    mass_flow_throat[time] = mass_flow[throat]

    dv_dt_throat[time] = 0.5*(dv_dt_p[throat] + dv_dt_c[throat])
    drho_dt_throat[time] = 0.5*(drho_dt_p[throat] + drho_dt_c[throat])

    total_time += dt
    if(time == 50):
        mass_flow_50[:] = mass_flow[:]
    if(time == 100):
        mass_flow_100[:] = mass_flow[:]
    if(time == 150):
        mass_flow_150[:] = mass_flow[:]
    if(time == 200):
        mass_flow_200[:] = mass_flow[:]
    if (time == 700):
        mass_flow_700[:] = mass_flow[:]



fig = plt.figure(figsize=(8,8),dpi = 100)
ax = fig.add_subplot(111)
ax.plot(x[:],mass_flow_rate_initial[:],label = "initial mass flow rate")
ax.set_xlabel("x, distance in Nozzle")
ax.set_ylabel("Mass Flow Rate(initial)")


fig = plt.figure(figsize=(15,4),dpi = 100)
ax = fig.add_subplot(111)
ax.plot(np.linspace(1,nt,nt),mass_flow_throat[:], label = "mass flow rate (throat)",linewidth = 2)
ax.set_xlabel("timesteps")
ax.set_ylabel("Mass flow in throat")
plt.rc('axes', labelsize = 20)
ax.grid()

fig = plt.figure(figsize=(15,4),dpi = 100)
ax = fig.add_subplot(111)
ax.plot(np.linspace(1,nt,nt),rho_throat[:],label="density in throat", linewidth = 2)
ax.set_xlabel("Timesteps")
ax.set_ylabel(r"Non-dimensional density $\frac{\rho}{\rho_0}$")
plt.rc('axes', labelsize = 17)
ax.grid()

fig = plt.figure(figsize=(15,5),dpi = 100)
ax = fig.add_subplot(111)
ax.plot(np.linspace(1,nt,nt),temp_throat[:],label = "temp. in throat", linewidth = 2)
ax.set_xlabel("Timesteps")
ax.set_ylabel(r"Non-dimensional Temperature at throat $\frac{T}{ T_0}$")
plt.rc('axes', labelsize = 13)
ax.grid()


fig = plt.figure(figsize=(15,5),dpi = 100)
ax = fig.add_subplot(111)
ax.plot(np.linspace(1,nt,nt),pressure_throat[:],label = "pressure in throat", linewidth = 2)
ax.set_xlabel("Timesteps")
ax.set_ylabel(r"Non-dimensional Pressure in Throat, $\frac{P}{P_0}$")
plt.rc('axes', labelsize = 13)
ax.grid()

fig = plt.figure(figsize=(15,5),dpi = 100)
ax = fig.add_subplot(111)
ax.plot(np.linspace(1,nt,nt),mach_number_throat[:],linewidth = 2)
ax.set_xlabel("Timesteps")
ax.set_ylabel(r"Mach Number in Throat, $M^{*}$")
plt.rc('axes', labelsize = 16)
ax.grid()

fig = plt.figure(figsize=(12,12), dpi = 100)
ax = fig.add_subplot(111)
ax.plot(x[:], rho[:], label = 'density', linewidth = 1, color = 'red')
ax2 = ax.twinx()
ax2.plot(x[:], mach_number[:], label = 'mach number',linewidth = 1,color = 'blue')
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=9,prop={'size': 15})
ax.set_xlabel("Non-Dimensional distance, x")
ax.set_ylabel(r"Non Demensional density  $\frac{\rho}{\rho_o}$")
ax2.set_ylabel("Mach Number")
ax.grid()
plt.show()