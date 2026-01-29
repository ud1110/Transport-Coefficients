import numpy as np
from scipy import integrate

class Conductivity:
    """
    This is a class for defining all the functions required to find the conductivity of the system.
    Class becomes essential when we need to run multiple simulations together (multiple codes can use one class)
    __init__: For defining the important parameters which we might need later, and to provide the important system parameters also (from instance)
    vel: For extracting all the important velocities
    cross_corr: For finding the general correlation function
    CF: For finding the correlation functions for our system
    cond: For finding the conductivity
    gnrl: A general function which can do everything, by calling other functions
    """
    
    def  __init__(self, v_cation, v_anion, V, times, T=298):
        self.V=V
        self.times=times
        self.T=T
        self.v_cation=v_cation
        self.v_anion=v_anion
        global kB
        kB=1.38*10**(-23)

    def vel(self, filename):
        """
        The way we are extracting parameters depends on the way they are stored in the file
        v=[]: Defines the array to be used for storing the sum of velocities at one timstamp (has size # timestamps * 3)
        vx, vy, vz: Sum of x-, y-, and z-velocities of all ions at one time
        Reading the file:
        ITEM: ATOMS id vx vy vz: This defines the start of velocities of new dataset. Upon encountering this, save previous velocities. Set flag=1
        ITEM: TIMESTEP: Defines the start of a new timestep, but has no data to be saved
        flag: Variable used to define when to store data and when to neglect (+1: store, -1: neglect)
        """
        v=[]
        vx=0
        vy=0
        vz=0
        flag=-1 #Initially (on the first line), there are no velocities, so nothing is to be collected

        with open(filename, 'r') as f: #'r': To read the file
            for line in (f):
                if "ITEM: ATOMS id vx vy vz" in line:
                    v.append([vx, vy, vz])
                    vx=0
                    vy=0
                    vz=0
                    flag=1
                elif "ITEM: TIMESTEP" in line:
                    flag=-1
                elif flag==1: #Between "ITEM: ATOMS id vx vy vz" and "ITEM: TIMESTEP", flag remains 1 (and we have data in this part only)
                    data=line.split("\n")[0].split(" ")[1:]
                    """
                    line: to read the line
                    .split("\n"): split when you see a new line
                    0: the text part of line
                    .split(" "): split the data upon seeing a space
                    1: 0th position is occupied by the Atom ID, so ignore it and move to next indices
                    """
                    vx+=float(data[0])
                    vy+=float(data[1])
                    vz+=float(data[2])
        v=np.array(v) #better to convert it to a numpy array
        return v

    def cross_corr(self, ja, jb):
        """
        Find the correlation function (first in the frequency domain (periodogram), and then transform it to the time domain, and then divide by n)
        fft: Computes the FFT, with X=fft(x, n); X_k=sum over j (x_j exp(-i*2pi/n * k*j))
        ifft: Computes the inverse FFT, with x=ifft(X); x_k=1/n sum over j (x_j exp(i*2pi/n *j*k))
        Cross Correlation Theorem: FT(sum over tau(x(t+tau)*y(t+tau)))=FT(x(t))*(FT(y(t)).conjugate())
        Padding is done to avoid circular correlations (which lead to negative lags), and FFT leads to circularity if padding isn't done properly
        fft is fastest when we have 2^integer number of elements in the array
        """
        N=len(ja)
        Fa=np.fft.fft(ja, 2**((2*N-1).bit_length()))
        Fb=np.fft.fft(jb, 2**((2*N-1).bit_length()))
        periodogram=Fa*(Fb.conjugate())
        sum_of_prod=np.fft.ifft(periodogram)
        required=sum_of_prod[:N].real #Real part of first N elements is required, ith element corresponds to a lag=i
        #Positive lags only: lag=0 comes N times, lag=1 comes N-1 times, lag=i comes N-i times
        n=N*np.ones(N)-np.arange(0, N)
        CF=required/n
        return CF

    def A_C_CF(self, v_anion, v_cation):
        """
        To determine the Auto/ Cross Correlation Function, given the velocities of cations and anions
        Pass the velocities (x-, y-, and z-) into the cross_corr function and then determine the CFs for a particular time and direction
        """
        N=len(v_anion) #v is the array of the average velocities, and its length is the number of timestamps
        acf_minus_minus=np.zeros((N, 3)) #3 for 3 directions
        acf_plus_plus=np.zeros((N, 3))
        ccf_plus_minus=np.zeros((N, 3))

        for i in range(3):
            acf_minus_minus[:, i]=self.cross_corr(v_anion[:, i], v_anion[:, i]) #To call another function, we need to use self.
            acf_plus_plus[:, i]=self.cross_corr(v_cation[:, i], v_cation[:, i])
            ccf_plus_minus[:, i]=self.cross_corr(v_cation[:, i], v_anion[:, i])
        return acf_minus_minus, acf_plus_plus, ccf_plus_minus

    def tpt_coeff(self, acf_minus_minus, acf_plus_plus, ccf_plus_minus):
        """
        From the GK-LRT, L_ab=V/kT integral(<ja(t)jb(0)>dt)=V/kT*integral A_C_CF dt (In our case, actually it's 1/(VkT) int A_C_CF dt and then avg)
        If we have N data points, we'll have N-1 bins in which we can do the integration
        For integration, we'll use the trapezoidal rule, Area=1/2*sum of parallel sides*height 
        height is assumed to be a constant, as per out dump file, and cumtrapz ensures that it's cumulative integral
        Our L returned would be L as a function of time, it would be an array and has the units 1/(length*time), length=Angstrom, time=fs
        """
        T=self.T
        times=self.times
        V=self.V
        N=len(acf_minus_minus)-1

        L_minus_minus=np.zeros((N, 3))
        L_plus_plus=np.zeros((N, 3))
        L_plus_minus=np.zeros((N, 3))

        for i in range(3):
            L_minus_minus[:, i]=1/(V*kB*T) * integrate.cumulative_trapezoid(acf_minus_minus[:, i], times) * (1/(10**(-10) * 10**(-15)))
            L_plus_plus[:, i]=1/(V*kB*T) * integrate.cumulative_trapezoid(acf_plus_plus[:, i], times) * (1/(10**(-10) * 10**(-15)))
            L_plus_minus[:, i]=1/(V*kB*T) * integrate.cumulative_trapezoid(ccf_plus_minus[:, i], times) * (1/(10**(-10) * 10**(-15)))

        L_m_m=np.mean(L_minus_minus, axis=1) #Spatial averaging
        L_p_p=np.mean(L_plus_plus, axis=1)
        L_p_m=np.mean(L_plus_minus, axis=1)
        return L_m_m, L_p_p, L_p_m

    def finalCalc(self):
        """
        This function is necessary so that we can call other functions from here.
        Trying to access, say cross_corr from ipynb using instance.A_C_CF isn't possible. A_C_CF can no longer access that unless we've self
        So, in this section, we shall specifically focus on that part, using self.function
        """
        vc=self.vel(self.v_cation)
        va=self.vel(self.v_anion)
        acf_m_m, acf_p_p, ccf_p_m=self.A_C_CF(va, vc)
        L_m_m, L_p_p, L_p_m=self.tpt_coeff(acf_m_m, acf_p_p, ccf_p_m)
        return acf_m_m, acf_p_p, ccf_p_m, L_m_m, L_p_p, L_p_m