from scipy.optimize import minimize, basinhopping
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


#these values can scale kinetic parameters as employed in Reaction.__init__
Vscalar = 10#5E-4    #scaling factor for VmB and VmG, increase for faster simulations, decrease for realistic stabilization times
Bscalar = 1
Gscalar = 1
Fscalar = 1

VmB= 14             #Vmax BOD       see main publication for explanation of below values
KfB= 3.21E-3        #Km FcMe2 BOD
KsB= 34.2E-3        #Km O2 BOD
VmG= 1              #Vmax GDH
KfG= 3.21E-4        #Km FcMe2+ GDH
KsG= 3.8E-3         #Km Glucose GDH
F0= 21.8E-3         #initial concentration of FcMe2

#initial guess and bounds for optimized parameters
def genRange(aValue):
    return [i / 10 * aValue for i in [10, 100]]
parNames         = {'VmB': VmB, 'KfB': KfB, 'KsB': KsB, 'VmG': VmG, 'KfG': KfG, 'KsG': KsG}
parRanges        = {parName: genRange(parValue) for parName, parValue in parNames.items()}
#only VmB is allowed to scale up, because scaling both often results in failure for V values to converge
parRanges['VmB'] = [VmB, VmB * 1.4]
parRanges['VmG'] = [VmG, VmG * 1.01]
bounds           = [(parRanges[key][0], parRanges[key][1]) for key in parNames]
initial_guess    = [bound[0] for bound in bounds]
total_calls      = 20   #number of iterations for Basin-Hopping

class Reaction:
    def __init__(self,VmB,KfB,KsB,VmG,KfG,KsG,F0,Vscalar,Bscalar,Gscalar,Fscalar, outputProcess = False):

        self.VmB = VmB * Vscalar * Bscalar
        self.KfB = KfB           * Bscalar * Fscalar
        self.KsB = KsB           * Bscalar
        self.VmG = VmG * Vscalar * Gscalar
        self.KfG = KfG           * Gscalar * Fscalar
        self.KsG = KsG           * Gscalar
        self.F0  = F0                      * Fscalar

        self.E0  = 0.2     #std redox potential of FcMe2, based on CV data

        # Simulation parameters
        self.iterTime     = 1E-4 * Fscalar/Vscalar                   #seconds to run reaction per iteration
        self.maxStabMins  = 10                                       #maximum time for stabilization at any SG, can be set arbitrarily
        self.maxStabIters = self.maxStabMins * 60 / self.iterTime    #maximum iterations for stabilization at any SG
        self.maxDV        = 1.001                                    #maximum DV to consider "stabilized" 1/maxDV < VG/VB < maxDV
        self.minDV        = 1/self.maxDV

        #iteration numbers to output checkpoints for long stabilizations
        self.outputProcess          = outputProcess  #in single runs, outPutProcess = True to see stabilization progress
        self.iterationCheckpoints   = np.linspace(0, self.maxStabIters, num=10).astype(int)

        #substrate concentrations
        self.SB         = 0.0013                                    #concentration of O2
        self.SGnum      = 11                                        #number of glucose concentrations to generate
        self.logSGs     = np.linspace(-7.5,-2.3,num=self.SGnum)     #logarithmic glucose concentrations
        self.SGs        = np.e**self.logSGs                         #glucose concentrations

        #for faster claculations
        self.KsBoSB = self.KsB / self.SB                            #precalulating KO2/O2 since it doesn't change
        self.VmGiter = self.VmG * self.iterTime                     #precalculating max VG since numerator is constant
        self.VmBiter = self.VmB * self.iterTime                     #as above

        #plotting parameters
        self.skipOCPs = 0           #point OCPs to skip before plotting
        self.xAxisLabel = 'sec'     #x-axis label for plotting
                                    #time conversion factor for x-axis label
        self.timeConversion = {'h': self.iterTime/3600, 'min': self.iterTime/60, 'sec': self.iterTime}[self.xAxisLabel]
        self.timeAxis       = np.array([]) # time axis for OCP values generated in trim_results
        self.pOCPTimeAxis   = np.array([]) # time axis for point OCP values generated in trim_results

        # Initialize lists
        self.Fs = []
        self.Fps = []
        self.VBs = []
        self.VGs = []
        self.pointOCPs  = []
        self.OCP        = []
        self.OCPiterations  = []

    def identify(self):
        """used to indentify parameters belonging to any randomly generated simulation"""
        self.values = {'VmB': self.VmB, 'KfB': self.KfB, 'KsB': self.KsB, 'VmG': self.VmG,
                       'KfG': self.KfG, 'KsG': self.KsG}
        global parNames
        factors = {key: self.values[key]/parNames[key] for key in self.values}
        for key in factors:
            print(f'{key}: {factors[key]:.2f}', end='\t')

    def simulate(self):
        def run(F, SG):
            """ run VB and VG for iterTime, calculate resulting values for F and Fp
                VB = dO2/dt = -dF/4dt
                VG = dG/dt  =  dF/2dt """
            Fp      = self.F0 - F                                                           #FcMe2+ conc
            VB      = self.VmBiter / (self.KfB / F + self.KsBoSB + 1)                       #BOD velocity
            VG      = self.VmGiter / (self.KfG / Fp + self.KsG / SG + 1) if SG != 0 else 0  #GDH velocity
            F_new   = F + 2 * VG - 4 * VB                                                   #new concentration of FcMe2
            dV      = 2*VB/VG                                                               #velocity difference to determine stabilization

            return F_new, VB, VG, dV
        def trim_results(iteration):
            """reformat results for calculations and plotting"""

            # Trim arrays
            skipSize            = self.OCPiterations[self.skipOCPs]
            self.Fs             = np.array(self.Fs[skipSize:])
            self.VBs            = np.array(self.VBs[skipSize:])
            self.VGs            = np.array(self.VGs[skipSize:])
            self.pointOCPs      = self.pointOCPs[self.skipOCPs:]        # OCP values collected when V values stabilize
            fullFoFs            = (self.F0 - self.Fs) / self.Fs         # full F+/F values for all iterations
            self.OCP            = self.E0 + 0.0252 * np.log(fullFoFs)   # OCP values calculated across full range via above
            self.SGs            = self.SGs[self.skipOCPs:]
            self.logSGs         = self.logSGs[self.skipOCPs:]
            self.pointOCPs      = self.pointOCPs[self.skipOCPs:]        # OCP values collected when V values stabilize

            # calculate time axis
            iterations          = np.linspace(skipSize, iteration + 1, num=iteration + 1 - skipSize)
            self.timeAxis       = iterations * self.timeConversion              # time axis for OCP values
            self.OCPiterations  = np.array(self.OCPiterations[self.skipOCPs:])  # used to calculate a full x-axis for stabilized OCP values
            self.pOCPTimeAxis   = self.OCPiterations[self.skipOCPs:] * self.timeConversion

        def fit():
            """fit a linear function of the OCP values generated in this reaction's simulation vs the input glucose concentrations"""

            def linear_fit(x, y, xterm):

                m, b = np.polyfit(x, y, 1)          # Fit a linear function to the data
                linear_equation = [m, b]                 # Define the linear function from the fit
                y_pred = m * x + b                       # Calculate the predicted y values
                r2 = r2_score(y, y_pred)

                return linear_equation + [r2]

            try:
                m, b, r2 = linear_fit(self.logSGs, self.pointOCPs, ' ln[G]')
            except Exception as e:
                raise ValueError(
                    f'SGsLen: {len(self.SGs)}, logSGsLen: {len(self.logSGs)}, pointOCPsLen: {len(self.pointOCPs)},\n {e}')

            self.m = m
            self.b = b
            self.r2 = r2
            mscore = abs(m + 0.0252) / 0.0252
            bscore = abs(b - 0.154) / 0.154
            score = mscore + bscore
            return score

        #initialize simulation parameters
        F = self.F0 - 1E-12                   #assume polymer starts as FcMe2
        iteration = -1

        #simulate
        for i, SG in enumerate(self.SGs):                               #for glucose concentration
            dV = np.inf                                                     #initialize V difference
            stabIter = -1                                                   #initialize stabilization iteration
                                                                            #an "iteration" is a time step of iterTime, e.g. dF produced by GDH activity in one iteration = VG * iterTime
            while not (self.minDV <= dV <= self.maxDV):                     #until Vs converge
                iteration += 1                                                  #increment iteration
                stabIter += 1                                                   #increment stabilization iteration
                F, VB, VG, dV = run(F, SG)                                      #run simulation for iterTime
                VB_F = 4 * VB                                                   #change in F from VB
                VG_F = 2 * VG                                                   #change in F from VG

                #update lists
                self.Fs.append(F)
                self.VBs.append(VB_F)
                self.VGs.append(VG_F)

                #if long stabilization, output 10 times before failing/completing
                if SG != self.SGs[0] and stabIter in self.iterationCheckpoints[1:]:
                    stabTime = stabIter * self.timeConversion
                    if self.outputProcess:
                        print(f'stabilizing dV to {self.maxDV:.2e}:'
                              f' {int(stabIter/self.maxStabIters*10)}/10,'
                              f' dV = {dV:.2e},\t time: {stabTime:.2f} {self.xAxisLabel}')

                #if failed to converge velocities
                if SG != self.SGs[0] and stabIter > self.maxStabIters:
                    if self.outputProcess:
                        print(f'stabilization time exceeded {self.maxStabMins} minutes.')
                    return 100 #high default output to indicate failure

                #if velocities converged successfully
                if (self.minDV <= dV <= self.maxDV):
                    FpoF = (self.F0 - F) / F
                    OCP = + self.E0 + 0.0252 * np.log(FpoF)                                #point OCP
                    self.pointOCPs      .append(OCP)                                       #append to pointOCPS
                    self.OCPiterations  .append(iteration)                                 #append iteration to plot pointOCPS
                    if self.outputProcess:
                        time = iteration * self.timeConversion          # calculate time
                        stabTime = stabIter * self.timeConversion       # calculate stabilization time
                        print(f'{i + 1}/{len(self.SGs)}\tSG: {SG:.4f},\tOCP: {OCP:.2e},'
                              f'\tdV: {dV:.2e}, \t time: {time:.2e} {self.xAxisLabel}'
                              f'\tstabTime: {stabTime:.2e} {self.xAxisLabel}')

        #after checking every substrate conc
        trim_results(iteration)
        score   = fit()
        return score

    def plot(self):
        print('plotting...')
        fig, ax1    = plt.subplots(figsize=(8, 6))


        # second legend
        sB = r"$_{\mathrm{BOD}}$"
        sGD = r"$_{\mathrm{GDH}}$"

        # Set axis labels
        ax1.set_xlabel(f'time / {self.xAxisLabel}', fontsize=20)
        ax1.set_ylabel('OCP / V vs SCE', color='orange', fontsize=20)
        ax1.tick_params(axis='y', labelcolor='orange')

        # add secondary axis for VB and VG
        ax2 = ax1.twinx()
        ax2.set_ylabel(r"$dM dt^{\mathrm{-1}}$ / "+u"\u00B1\u03bc"+"Mol s$^{-1}$ FcMe$_2$", color='black', fontsize=15)
        ax2.tick_params(axis='y', labelcolor='black')

        #plot data
        ax1.plot(self.timeAxis, self.OCP, color='orange', label = 'OCP')  # OCP plot
        #ax1.plot(self.pOCPTimeAxis, self.pointOCPs, marker='.', color='purple', label = "final OCP", linestyle='none')

        self.VBs *= 1E6
        self.VGs *= 1E6

        ax2.plot(self.timeAxis, self.VBs, label=f'-4 * V{sB}', linestyle=':', color='green')
        ax2.plot(self.timeAxis, self.VGs, label=f' 2 * V{sGD}', linestyle='--', color='red', alpha=0.5)


        plt.show()
    def evaluate(self):
        """print the linear fit of OCP-Glucose for this reaction"""
        print(f'{self.m:.3f}ln(G) + {self.b:.3f}, r2 = {self.r2}', end = ' ')

def objective(params):
    global current_call, prevStartTime, outs

    # Track progress
    current_call += 1
    thisStartTime = time.time()
    prevStartTime = thisStartTime

    # Evaluate the Reaction simulation
    reaction    = Reaction(*params, F0, Vscalar, Bscalar, Gscalar, Fscalar)
    score       = reaction.simulate()
    outs.append(score)
    return score

class RandomDisplacement:
    """class defining steps between each basin hop"""
    def __init__(self, step_size=0.1, bounds=None):
        self.step_size = step_size
        self.bounds    = bounds

    def __call__(self, x):
        # Add a random displacement to each parameter
        new_x = np.empty_like(x)
        for i, val in enumerate(x):
            low, high = self.bounds[i]
            param_range = high - low
            # Scale the uniform step by a fraction of the allowed range
            step = np.random.uniform(-self.step_size * param_range, self.step_size * param_range)
            new_x[i] = val + step
        return new_x

# Basin-Hopping optimizer
outs = []
current_call = -1
random_step      = RandomDisplacement(step_size=0.1, bounds=bounds)  # Small step size for fine exploration
minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds, "options": {"maxiter": total_calls}}

# Run the Basin-Hopping algorithm
result = basinhopping(
    objective,
    x0               = initial_guess,
    minimizer_kwargs = minimizer_kwargs,
    niter            = total_calls,
    take_step        = random_step,
    disp             = True,
)

# Plotting results of optimization
def plotOutput(outs, result):
    plt.figure(figsize=(10, 5))
    plt.plot([i if i < 5 else 5 for i in outs], marker='o')
    plt.title('Objective Function Value vs. Iteration Number')
    plt.xlabel('Iteration Number')
    plt.ylabel('Objective Function Value')
    plt.grid(True)
    plt.show()

    # Output the best results
    print("Best parameters found:")
    for name, val in zip(parNames, result.x):
        rangeWarn   = f'{val / parNames[name]:.2f}'
        outVal      = f'{val:.2e} #{rangeWarn}'
        print(f"{name}= {outVal}")

    print(f"Lowest score: {result.fun}")
    reaction = Reaction(*result.x, F0, Vscalar, Bscalar, Gscalar, Fscalar)
    reaction.simulate()
    reaction.evaluate()
plotOutput(outs, result)

#example of a single simulation
def example_single_sim():
    global Vscalar, Bscalar, Gscalar, Fscalar, F0
    VmB = 1.40e+01  # 1.00x
    KfB = 1.58e-02  # 4.92x
    KsB = 3.50e-02  # 1.02x
    VmG = 1.00e+00  # 1.00x
    KfG = 3.25e-04  # 1.01x
    KsG = 3.80e-03  # 1.00x
    pars = [VmB, KfB, KsB, VmG, KfG, KsG]
    a = Reaction(*pars, F0, Vscalar, Bscalar, Gscalar, Fscalar, outputProcess = True)
    a.simulate()
    a.plot()
    a.evaluate()
example_single_sim()
