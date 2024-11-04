# add path to and import custom qmeq
import sys
sys.path.append('../qmeq/')
import qmeq

# other imports
import numpy as np
from scipy.integrate import quad

# load plotting libraries
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# for parallelelization
#import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

# for progress bar
from tqdm import tqdm as tqdm

# define a class whose properties are the parameters of the system
class Parameters:
    """
    Class to store the parameters of the system.
    """
    def __init__(self, B=0., u=100., Tl=1., Tr=1., gamma=0.1, kerntype='RTDnoise', dband=1e4, countingleads=[0,2]):
        """
        Initialize the Parameters object.

        Parameters:
        - vg (float): Gate voltage.
        - vb (float): Bias voltage.
        - B (float): Magnetic field.
        - u (float): Coulomb interaction strength.
        - Tl (float): Temperature of the left lead.
        - Tr (float): Temperature of the right lead.
        - gamma (float): Coupling strength between leads and dots.
        - kerntype (str): Type of kernel used for calculations.
        - dband (float): Energy band width.
        - countingleads (list): Leads for counting statistics.
        """
        self.B = B
        self.u = u
        self.Tl = Tl
        self.Tr = Tr
        self.gamma = gamma
        self.kerntype = kerntype
        self.dband = dband
        self.countingleads = countingleads

# define results class
class Results:
    """
    Class to store the results of the stability diagram calculation.
    """
    def __init__(self,results,Vg,Vb,params):
        """
        Initialize the Results object.

        Parameters:
        - I: Current values.
        - S: Noise values.
        - Q: Heat current values.
        """

        ng, nb = len(Vg), len(Vb)

        self.I = np.array([r[0].real for r in results[:,0]]).reshape(ng,nb)
        self.S = np.array([r[1].real for r in results[:,0]]).reshape(ng,nb)
        self.I_first = np.array([r[0].real for r in results[:,1]]).reshape(ng,nb)
        self.S_first = np.array([r[1].real for r in results[:,1]]).reshape(ng,nb)
        self.Q0 = np.array([r[0] for r in results[:,3]]).reshape(ng,nb)
        self.Q1 = np.array([r[1] for r in results[:,3]]).reshape(ng,nb)
        self.Q2 = np.array([r[2] for r in results[:,3]]).reshape(ng,nb)
        self.Q3 = np.array([r[3] for r in results[:,3]]).reshape(ng,nb)
        self.Istd0 = np.array([r[0] for r in results[:,2]]).reshape(ng,nb)
        self.Istd1 = np.array([r[1] for r in results[:,2]]).reshape(ng,nb)
        self.Istd2 = np.array([r[2] for r in results[:,2]]).reshape(ng,nb)
        self.Istd3 = np.array([r[3] for r in results[:,2]]).reshape(ng,nb)
        self.Vg = Vg
        self.Vb = Vb
        self.params = params
        
# define results class
class Results_p:
    """
    Class to store the results of the stability diagram calculation.
    """
    def __init__(self,results,Vg,Vb,params):
        """
        Initialize the Results object.

        Parameters:
        - I: Current values.
        - S: Noise values.
        - Q: Heat current values.
        """

        ng, nb = len(Vg), len(Vb)

        self.I = np.array([r[0].real for r in results[:,0]]).reshape(ng,nb)
        self.S = np.array([r[1].real for r in results[:,0]]).reshape(ng,nb)
        self.Q0 = np.array([r[0] for r in results[:,2]]).reshape(ng,nb)
        self.Q1 = np.array([r[1] for r in results[:,2]]).reshape(ng,nb)
        self.Q2 = np.array([r[2] for r in results[:,2]]).reshape(ng,nb)
        self.Q3 = np.array([r[3] for r in results[:,2]]).reshape(ng,nb)
        self.Istd0 = np.array([r[0] for r in results[:,1]]).reshape(ng,nb)
        self.Istd1 = np.array([r[1] for r in results[:,1]]).reshape(ng,nb)
        self.Istd2 = np.array([r[2] for r in results[:,1]]).reshape(ng,nb)
        self.Istd3 = np.array([r[3] for r in results[:,1]]).reshape(ng,nb)
        self.Vg = Vg
        self.Vb = Vb
        self.params = params

def anderson(vg=0.,vb=0.,parameters=Parameters()):
    """
    Build Anderson model with 4 leads and 2 dots.
    
    Parameters:
    - vg (float): Gate voltage.
    - vb (float): Bias voltage.
    - B (float): Magnetic field.
    - u (float): Coulomb interaction strength.
    - Tl (float): Temperature of the left lead.
    - Tr (float): Temperature of the right lead.
    - gamma (float): Coupling strength between leads and dots.
    - kerntype (str): Type of kernel used for calculations.
    - dband (float): Energy band width.
    - countingleads (list): Leads for counting statistics.
    
    Returns:
    - system (qmeq.Builder): Anderson model system.
    """
    # extract parameters from parameters object
    B = parameters.B
    u = parameters.u
    Tl = parameters.Tl
    Tr = parameters.Tr
    gamma = parameters.gamma
    kerntype = parameters.kerntype
    dband = parameters.dband
    countingleads = parameters.countingleads
    
    n = 2
    h = {(0,0):-vg+B, (1,1):-vg-B}
    U = {(0,1,1,0):u}

    nleads = 4
    mulst = {0:vb/2, 1:-vb/2, 2:vb/2, 3:-vb/2}
    tlst = {0:Tl, 1:Tr, 2:Tl, 3:Tr}

    t = np.sqrt(gamma/np.pi/2)
    tleads = {(0, 0):t, (1, 0):t, (2, 1):t, (3, 1):t}

    system = qmeq.Builder(nsingle=n, hsingle=h, coulomb=U, nleads=nleads, 
                        mulst=mulst, tlst=tlst, tleads=tleads, dband=dband,
                        countingleads=countingleads, kerntype=kerntype,itype=1)
    
    return system

def solve_bias_gate_p(vg,vb,params):
    """
    Set the bias and gate voltages for a given system.

    Parameters:
    system (object): The system object representing the physical system.
    vg (float): Gate voltage.
    vb (float): Bias voltage.
    """
    system=anderson(vg,vb,params)
    system.solve()
    
    return system.current_noise, system.current, system.heat_current

def solve_bias_gate(vg,vb,params):
    """
    Set the bias and gate voltages for a given system.

    Parameters:
    system (object): The system object representing the physical system.
    vg (float): Gate voltage.
    vb (float): Bias voltage.
    """
    system=anderson(vg,vb,params)
    system.solve()
    
    return system.current_noise, system.current, system.heat_current, system.appr.current_noise_o4trunc

def solve_bias_gate_std(vg,vb,params):
    """
    Set the bias and gate voltages for a given system.

    Parameters:
    system (object): The system object representing the physical system.
    vg (float): Gate voltage.
    vb (float): Bias voltage.
    """
    system=anderson(vg,vb,params)
    system.solve()
    
    return system.current, system.heat_current

def pstab(Vg,Vb,params):
    """
    Calculate the current and heat current for a given set of parameters.

    Parameters:
    Vg (array-like): Array of gate voltages.
    Vb (array-like): Array of bias voltages.
    params (object): The parameters of the anderson model.

    Returns:
    tuple: A tuple containing the current, noise, and heat current arrays.
    """
    # prepare results array
    with ProcessPoolExecutor() as executor:
        results = np.array(list(tqdm(executor.map(solve_bias_gate, np.tile(Vg,len(Vb)), np.repeat(Vb,len(Vg)), [params]*len(Vg)*len(Vb)),total=len(Vg)*len(Vb))),dtype=object)
        
    return results

def pstab_std(Vg,Vb,params):
    """
    Calculate the current and heat current for a given set of parameters.

    Parameters:
    Vg (array-like): Array of gate voltages.
    Vb (array-like): Array of bias voltages.
    params (object): The parameters of the anderson model.

    Returns:
    tuple: A tuple containing the current, noise, and heat current arrays.
    """
    # prepare results array
    with ProcessPoolExecutor() as executor:
        results = np.array(list(tqdm(executor.map(solve_bias_gate_std, np.tile(Vg,len(Vb)), np.repeat(Vb,len(Vg)), [params]*len(Vg)*len(Vb)),total=len(Vg)*len(Vb))),dtype=object)
        
    return results

def pstab_p(Vg,Vb,params):
    """
    Calculate the current and heat current for a given set of parameters.

    Parameters:
    Vg (array-like): Array of gate voltages.
    Vb (array-like): Array of bias voltages.
    params (object): The parameters of the anderson model.

    Returns:
    tuple: A tuple containing the current, noise, and heat current arrays.
    """
    # prepare results array
    with ProcessPoolExecutor() as executor:
        results = np.array(list(tqdm(executor.map(solve_bias_gate_p, np.tile(Vg,len(Vb)), np.repeat(Vb,len(Vg)), [params]*len(Vg)*len(Vb)),total=len(Vg)*len(Vb))),dtype=object)
        
    return results

def psweep(Vg,Vb,params):
    """
    Calculate the current and heat current for a given set of parameters.

    Parameters:
    Vg (array-like): Array of gate voltages.
    Vb (array-like): Array of bias voltages.
    params (object): The parameters of the anderson model.

    Returns:
    tuple: A tuple containing the current, noise, and heat current arrays.
    """
    if len(Vg) != len(Vb):
        raise ValueError("Vg and Vb must have the same length")

    # prepare results array
    with ProcessPoolExecutor() as executor:
        results = np.array(list(tqdm(executor.map(solve_bias_gate, Vg, Vb, [params]*len(Vg)),total=len(Vg))),dtype=object)
        
    return results

# plot current, noise, heat current and TUR with imshow and colorbar
def plot_stab(I, S, Vg, Vb, linthresh=1e-3):
    """
    Plot stability diagrams for current, noise, differential conductance, and noise derivative.

    Parameters:
    - I (2D array): Current values.
    - S (2D array): Noise values.
    - Vg (1D array): Gate voltage values.
    - Vb (1D array): Bias voltage values.
    - linthresh (float, optional): Threshold for linear scaling of noise derivative. Default is 1e-3.
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    plt.colorbar(axes[0, 0].imshow(I, origin='lower', aspect='auto', extent=[Vg[0], Vg[-1], Vb[0], Vb[-1]]), ax=axes[0, 0])
    axes[0, 0].set_xlabel('$V_g$')
    axes[0, 0].set_ylabel('$V_b$')
    axes[0, 0].set_title('Current')
    plt.colorbar(axes[0, 1].imshow(S, origin='lower', aspect='auto', extent=[Vg[0], Vg[-1], Vb[0], Vb[-1]]), ax=axes[0, 1])
    axes[0, 1].set_xlabel('$V_g$')
    axes[0, 1].set_ylabel('$V_b$')
    axes[0, 1].set_title('Noise')
    dI = np.gradient(I)[0]
    normI = colors.LogNorm(vmin=dI[dI > 0].min(), vmax=dI.max())
    plt.colorbar(axes[1, 0].imshow(dI, origin='lower', aspect='auto', extent=[Vg[0], Vg[-1], Vb[0], Vb[-1]], norm=normI), ax=axes[1, 0])
    axes[1, 0].set_xlabel('$V_g$')
    axes[1, 0].set_ylabel('$V_b$')
    axes[1, 0].set_title('Differential conductance dI/dV')
    dS = np.gradient(S)[0]
    linthresh = np.abs(dS).max() * 1e-3
    normS = colors.SymLogNorm(linthresh=linthresh, vmin=dS.min(), vmax=dS.max())
    plt.colorbar(axes[1, 1].imshow(dS, origin='lower', aspect='auto', extent=[Vg[0], Vg[-1], Vb[0], Vb[-1]], norm=normS), ax=axes[1, 1])
    axes[1, 1].set_xlabel('$V_g$')
    axes[1, 1].set_ylabel('$V_b$')
    axes[1, 1].set_title('Noise derivative dS/dV')
    plt.tight_layout()
    plt.show()

#### SRL ###
    
def fermi(e,mu,T):
    x = (e-mu)/T
    if x > 709.77:  # To avoid overflow in the denominator
        return 0.0
    return 1/(np.exp(x)+1)

def trans(e,gamma,vg):
    return gamma**2/((e+vg)**2+gamma**2)

def int_current(e,gamma,vg,TL,TR,vb):
    t = trans(e,gamma,vg)
    fL,fR = fermi(e,vb/2,TL),fermi(e,-vb/2,TR)
    return t*(fL-fR)

def int_noise(e,gamma,vg,TL,TR,vb):
    t = trans(e,gamma,vg)
    fL,fR = fermi(e,vb/2,TL),fermi(e,-vb/2,TR)
    return t*(fL*(1-fL)+fR*(1-fR))+ t*(1-t)*(fL-fR)**2

def int_energy(e,gamma,vg,TL,TR,vb):
    t = trans(e,gamma,vg)
    fL,fR = fermi(e,vb/2,TL),fermi(e,-vb/2,TR)
    return e*t*(fL-fR)

def SRL_exact(gamma,gate,bias,TL,TR,dband=1e2):
    # calculate current
    f = lambda e: int_current(e,gamma,gate,TL,TR,bias)
    I = quad(f,-dband,dband)[0]/(2*np.pi)
    # calculate noise
    g = lambda e: int_noise(e,gamma,gate,TL,TR,bias)
    S = quad(g,-dband,dband)[0]/(2*np.pi)
    # calculate energy current
    h = lambda e: int_energy(e,gamma,gate,TL,TR,bias)
    I_e = quad(h,-dband,dband)[0]/(2*np.pi)
    # heat current
    Q = I_e - bias/2*I
    return I,S,Q