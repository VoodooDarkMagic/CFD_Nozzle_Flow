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
Cx = 0.2

rho = np.zeros((nx))
T = np.zeros((nx))
A = np.zeros((nx))
v = np.zeros((nx))
P = np.zeros((nx))
mach_number = np.zeros((nx))
mass_flow = np.zeros((nx))

rho_throat = np.zeros((nt))
pressure_throat = np.zeros((nt))
velocity_throat = np.zeros((nt))
temp_throat = np.zeros((nt))
mach_number_throat = np.zeros((nt))
mass_flow_throat = np.zeros((nt))

#Initial profile
for i in range(nx):
    if x[i] <=0.5:
        rho[i] = 1
        T[i] = 1
    elif(x[i] > 0.5 and x[i] <= 1.5):
        rho[i] = 1 - 0.366*(x[i] - 0.5)
        T[i] = 1 - 0.167*(x[i] - 0.5)
    elif(x[i]>1.5 and x[i]<=2.1):
        rho[i] = 0.634 - 0.3879*(x[i]-1.5)
        T[i] = 0.833 - 0.3507*(x[i]-1.5)
    elif(x[i] > 2.1 and x[i] <= 3):
        rho[i] = 0.5892 + 0.10228*(x[i] - 2.1)
        T[i] = 0.93968 + 0.0622*(x[i] - 2.1)

    A[i] = 1 + 2.2*((x[i]-1.5)**2)
    v[i] = 0.59/(rho[i]*A[i])
    P[i] = rho[i]*T[i]
    mass_flow[i] = rho[i]*v[i]*A[i]
    
mass_flow_rate_initial = rho*v*A
throat = np.where(A == 1)
throat = throat[0]


#Initial Solution
U1 = np.zeros((nx))
U2 = np.zeros((nx))
U3 = np.zeros((nx))

U1_old = np.zeros((nx))
U2_old = np.zeros((nx))
U3_old = np.zeros((nx))

F1 = np.zeros((nx))
F2 = np.zeros((nx))
F3 = np.zeros((nx))
J = np.zeros((nx))

dU1_dt_p = np.zeros((nx))
dU2_dt_p = np.zeros((nx))
dU3_dt_p = np.zeros((nx))

dU1_dt_c = np.zeros((nx))
dU2_dt_c = np.zeros((nx))
dU3_dt_c = np.zeros((nx))

dU1_dt_av = np.zeros((nx))
dU2_dt_av = np.zeros((nx))
dU3_dt_av = np.zeros((nx))

S1 = np.zeros((nx))
S2 = np.zeros((nx))
S3 = np.zeros((nx))

for i in range(nx):
    U1[i] = rho[i]*A[i]
    U2[i] = rho[i]*A[i]*v[i]
    U3[i] = rho[i]*(T[i]/(gamma - 1) + (gamma/2)*(v[i]**2))*A[i]


for time in range(nt):

    U1_old = U1.copy()
    U2_old = U2.copy()
    U3_old = U3.copy()

    dt=np.min((0.5*dx)/(T**0.5 + v))
    print(dt)

    for i in range(nx):
        F1[i] = U2[i]
        F2[i] = ((U2[i]**2)/U1[i]) + (((gamma-1)/gamma))*(U3[i] - ((gamma/2)*((U2[i]**2)/U1[i])))
        F3[i] = (gamma*((U2[i]*U3[i])/U1[i])) - (gamma*(gamma-1)/2)*(U2[i]**3/U1[i]**2)
    
    for i in range(1,nx-1):
        S1[i] = Cx*(abs(P[i-1] - 2*P[i] + P[i+1])/(P[i-1] + 2*P[i] + P[i+1]))*(U1[i+1] - 2*U1[i] + U1[i-1])
        S2[i] = Cx*(abs(P[i-1] - 2*P[i] + P[i+1])/(P[i-1] + 2*P[i] + P[i+1]))*(U2[i+1] - 2*U2[i] + U2[i-1])
        S3[i] = Cx*(abs(P[i-1] - 2*P[i] + P[i+1])/(P[i-1] + 2*P[i] + P[i+1]))*(U3[i+1] - 2*U3[i] + U3[i-1])

    #Predictor step
    for i in range (1,nx-1):
        dA_dx = (A[i+1] - A[i])/dx
        J[i] = gamma_inverse*rho[i]*T[i]*dA_dx

        dU1_dt_p[i] = -((F1[i+1] - F1[i])/dx)
        dU2_dt_p[i] = -((F2[i+1] - F2[i])/dx) + J[i]
        dU3_dt_p[i] = -((F3[i+1] - F3[i])/dx)

        #solution update    
        U1[i] = U1[i] + dU1_dt_p[i]*dt + S1[i]
        U2[i] = U2[i] + dU2_dt_p[i]*dt + S2[i]#S2 
        U3[i] = U3[i] + dU3_dt_p[i]*dt + S3[i]#S3


    #getting the primitives
    for i in range(1,nx-1):
        rho[i] = U1[i]/A[i]
        v[i] = U2[i]/U1[i]
        T[i] = (gamma - 1)*((U3[i]/U1[i]) - ((gamma/2)*(v[i]**2)))
        P[i] = rho[i]*T[i]

    #updating F values
    for i in range(1,nx-1):
        F1[i] = U2[i]
        F2[i] = ((U2[i]**2)/U1[i]) + (((gamma-1)/gamma))*(U3[i] - ((gamma/2)*((U2[i]**2)/U1[i])))
        F3[i] = gamma*(U2[i]*U3[i]/U1[i]) - (gamma*(gamma-1)/2)*(U2[i]**3/U1[i]**2)

    #update S
    for i in range(1,nx-1):
        S1[i] = Cx*(abs(P[i-1] - 2*P[i] + P[i+1])/(P[i-1] + 2*P[i] + P[i+1]))*(U1[i+1] -2*U1[i] + U1[i-1])
        S2[i] = Cx*(abs(P[i-1] - 2*P[i] + P[i+1])/(P[i-1] + 2*P[i] + P[i+1]))*(U2[i+1] -2*U2[i] + U2[i-1])
        S3[i] = Cx*(abs(P[i-1] - 2*P[i] + P[i+1])/(P[i-1] + 2*P[i] + P[i+1]))*(U3[i+1] -2*U3[i] + U3[i-1])
    #Corrector Step
    for i in range(nx-2,0,-1):
        dA_dx = (A[i] - A[i-1])/dx
        J[i] = gamma_inverse*rho[i]*T[i]*dA_dx

        dU1_dt_c[i] = -((F1[i] - F1[i-1])/dx)
        dU2_dt_c[i] = -((F2[i] - F2[i-1])/dx) + J[i]
        dU3_dt_c[i] = -((F3[i] - F3[i-1])/dx)
    
       
        #Final solution update 
        U1[i] = U1_old[i] + 0.5*(dU1_dt_p[i] + dU1_dt_c[i])*dt + S1[i]#S1
        U2[i] = U2_old[i] + 0.5*(dU2_dt_p[i] + dU2_dt_c[i])*dt + S2[i] #S2
        U3[i] = U3_old[i] + 0.5*(dU3_dt_p[i] + dU3_dt_c[i])*dt + S3[i]#S3
    #Applying boundary condition

    #inlet
    U1[0] = rho[0]*A[0]
    U2[0] = 2*U2[1] - U2[2]
    U3[0] = U1[0]*((T[0]/(gamma-1)) + (gamma/2)*(v[0]**2))

    #outlet
    U1[nx-1] = 2*U1[nx-2] - U1[nx-3]
    U2[nx-1] = 2*U2[nx-2] - U2[nx-3]
    U3[nx-1] = ((P[nx-1]*A[nx-1])/(gamma - 1)) + (gamma/2)*(U2[nx-1])*(v[nx-1])

    
    rho = U1/A
    v = U2/U1
    T = (gamma - 1)*((U3/U1) - ((gamma/2)*(v**2)))
    P = rho*T
    mass_flow = rho*A*v
    mach_number = v/T**0.5


    #calculating variables in throat
    rho_throat[time] = rho[throat] 
    pressure_throat[time] = P[throat] 
    velocity_throat[time] = v[throat] 
    temp_throat[time] = T[throat] 
    mach_number_throat[time] = mach_number[throat] 
    mass_flow_throat[time] = mass_flow[throat]

fig = plt.figure(figsize=(10,10), dpi = 100)
ax = fig.add_subplot(111)
ax.plot(x[:], P[:],label = 'pressure', linewidth = 1.5)
ax.set_xlabel("Non-Dimensioal Distance x")
ax.set_ylabel("Non-Dimensional Pressure")
plt.rc('axes', labelsize = 15)
ax.legend(prop={'size': 15})

fig = plt.figure(figsize=(10,10), dpi = 100)
ax = fig.add_subplot(111)
ax.plot(x[:], mach_number[:],label = 'pressure', linewidth = 1.5)
ax.set_xlabel("Non-Dimensioal Distance x")
ax.set_ylabel("Mach Number")
plt.rc('axes', labelsize = 15)
ax.legend(prop={'size': 15})