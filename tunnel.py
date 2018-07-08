import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fp
import scipy.sparse as sp
import scipy.sparse.linalg as la
from functools import reduce

# The algorithm idea follows A Goldberg, HM Schey, JL Schwartz, 1967,
# Computer-generated motion pictures of one-dimensional quantum-mechanical transmission and reflection phenomena
# American Journal of Physics, vol. 35, p. 177,
# http://ergodic.ugr.es/cphys/lecciones/SCHROEDINGER/ajp.pdf
# I also attempted to include absorbing boundary condition 
# T. Fevens and H. Jiang. 1999, Absorbing Boundary Conditions for 
# the Schrodinger Equation, SIAM J. Sci. Comput., 21, p. 255.
# https://pdfs.semanticscholar.org/6ff7/893f6b4e5691b349d7f6bee138a9e472ff53.pdf
# The program uses units in which hbar=1.
# The program solves the relevant equations using sparse matrices and LU decomposition.
# For an explanation of the use of LU decomposition to solve linear equations, see
#Trefethen, Lloyd N.; Bau, David (1997), 
#Numerical linear algebra, Philadelphia: Society for Industrial and Applied Mathematics,
# Lecture 20.


def boundary_conditions2(dx,dt,pot,qs,m):
    [q1,q2] = qs
    [c1,c2] = [2/(q1+q2),q1*q2/(q1+q2)]
    [t1,t2] = [1j/(2*dx),-1j*c1/(2*dt)]
    vt = c1*c2*pot/4.0
    [p1,p2] = [t2-t1+vt,t2+t1+vt]
    [p3,p4] = [t2+t1-vt,t2-t1-vt]
    return [p1,p2,p3,p4]

def boundary_conditions3(dx,dt,pot,qs,m):
    [q1,q2,q3] = qs
    [h1,h3] = [m*(q1+q2+q3),m**3*(q1*q2*q3)]
    h2 = m**2*(q1*q2*q3)*((1/q1)+(1/q2)+(1/q3))
    [t1,t2] = [1j*(h2-pot)/(2*dx),1/(dt*dx)]
    [t3,t4] = [1j*h1/(2*dt),(h3-h2*pot)/4]
    [p1,p2] = [t2-t1-t3-t4,t1-t2-t3+t4]
    [p3,p4] = [t1+t2-t3+t4,t4-t1-t2-t3-t4]
    return [p1,p2,p3,p4]

def simulation_dic_setup(prepsimdic):
    [sigma,k0,initx] = map(lambda k:prepsimdic[k],['sigma','k0','initx'])
    [m,p,maxx] = map(lambda k:prepsimdic[k],['mass','p','maximum_x'])
    func_dic = {2:boundary_conditions2,3:boundary_conditions3}
    simdic = {k:i for (k,i) in prepsimdic.items()}
    simdic['boundary_condition_function'] = func_dic[p]
    simdic['energy'] = k0**2/(2*m)
    simdic['qs'] = map(lambda ind:k0/m,range(p))
    simdic['init_psi'] = lambda x: gaussian_wavepacket(x,sigma,initx,k0)
    simdic['maximum_t'] = simdic.get('maximum_t',2.0*maxx*m/k0)
    return simdic

def simulation_p_2_3(prepsimdic):
    simdic = simulation_dic_setup(prepsimdic)
    [numxs,numsteps,potfunc] = map(lambda k:simdic[k],['numxs','numsteps','potential_function'])
    [psi0func,maxx,maxt] = map(lambda k:simdic[k],['init_psi','maximum_x','maximum_t'])
    [m,bcfunc,qs] = map(lambda k:simdic[k],['mass','boundary_condition_function','qs']) 
    [dx,dt]=[(1.0*maxx)/numxs,(1.0*maxt)/numsteps]
    k0 = prepsimdic['k0']
    xs = map(lambda x:x*dx,range(numxs+1))
    psi0 = np.array(map(psi0func,xs))
    psis = [psi0]
    V = map(potfunc,xs)
    [left1,left2,left3,left4] =  bcfunc(dx,dt,V[0],qs,m)
    [right1,right2,right3,right4]  =  bcfunc(dx,dt,V[-1],qs,m)
    alpha = 1j*dt/(2*dx**2)
    xsis = map(lambda pot:1+1j*dt/2*(2/(dx**2)+pot),V)
    xsis[0] = left1
    xsis[-1] = right1
    gammas = map(lambda pot:1-1j*dt/2*(2/(dx**2)+pot),V)
    gammas[0] = left3
    gammas[-1] = right3
    alphau1upper = map(lambda x:-1*alpha,xs[1:])
    alphau1upper[0] = left2
    alphau1lower = map(lambda x:-1*alpha,xs[1:])
    alphau1lower[-1] = right2
    alphau2upper = map(lambda x:alpha,xs[1:])
    alphau2upper[0] = left4
    alphau2lower = map(lambda x:alpha,xs[1:])
    alphau2lower[-1] = right4
    diagnums = [1,0,-1]
    u1 = sp.diags([alphau1upper,xsis,alphau1lower],diagnums).tocsc()
    u2 = sp.diags([alphau2upper,gammas,alphau2lower],diagnums).tocsc()
    LU = la.splu(u1)
    for n in range(numsteps):
        currpsi = psis[-1]
        b = u2.dot(currpsi)
        nextpsi = LU.solve(b)
        psis.append(nextpsi)
    return {'wavefunction':psis,'xs':xs,'k0':k0,'mass':m,
            'maximum_x':prepsimdic['maximum_x'],
            'numxs':prepsimdic['numxs'],
            'potential':map(lambda p:p/simdic['energy'],V),
            'times':map(lambda n:n*dt,range(numsteps+1))}

def gaussian_wavepacket(x,sigma,initx,k0):
    front = (1/((sigma**2)*np.pi))**0.25
    back = np.exp(-1*(x-initx)**2.0/(2.0*(sigma**2.0))+1j*k0*x)
    return front * back

def in_window(x,start,end):
    return x > start and x < end

def step_potential(x,lower,upper,start,end):
    return upper if in_window(x,start,end) else lower
    
def multisteps_same_levels(x,lower,upper,starts,ends):
    return upper if any(map(lambda s,e:in_window(x,s,e), starts,ends)) else lower

def multisteps_different_levels(x,default,levels_windows):
    #level_window = [[s,e] level]
    levwind = filter(lambda lw: in_window(x,lw[0][0],lw[0][1]),levels_windows)
    return levwind[1] if levwind else default

def gaussian_potential(x,magnitude,sigma,initx):
    return magnitude*np.exp(-1*(x-initx)**2.0/(2.0*(sigma**2.0)))
    
def many_gaussian_potentials(x,sigmas,initxs):
    gatx = lambda m,s,ix:gaussian_potential(x,m,s,ix)
    return reduce((lambda x,y: x+y),map(gatx,sigmas,initxs))
  
sim_dic = {'numxs':2000,'numsteps':2000,
           'potential_function': lambda x: step_potential(x,0.0,110.0,50.0,50.5),
            'maximum_x':100,'mass':0.5,'sigma':5,'k0':10,'initx':10.0,'p':3}
            
# q = k0/m = groupvelocity, energy = k**2/2m, 

simres = simulation_p_2_3(sim_dic)

def plot_simulation_step(data,stepnum,resfunc,fn,c1,c2):
    plt.clf()
    fig, ax1 = plt.subplots()
    [xs,wfstep] = [data['xs'],data['wavefunction'][stepnum]]
    pot = data['potential']
    ax2 = ax1.twinx()
    ax1.plot(xs, resfunc(wfstep), color=c1)
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('square amplitude')
    ax2.plot(xs, pot, color=c2)
    ax2.set_ylim(0,max(pot)*1.1)
    ax2.set_ylabel('Potential (in units of wavefunction energy)')
    plt.savefig(fn+str(stepnum)+'.png')
    plt.close('all')

def square_amplitude(compnum):
    return np.abs(compnum)**2

def phase(compnum):
    return np.argument(compnum)

for ts in range(0,1001,100):
    plot_simulation_step(simres,ts,square_amplitude,'wavef','blue','red')

def fft_wavef_step(data,stepnum,fn):
    fdata = np.abs(fp.fft(data['wavefunction'][stepnum])[:400])
    binsize = (1.0*data['maximum_x'])/data['numxs']
    fs = map(lambda n:(n*binsize)**2,range(len(fdata)))[:400]
    [maxfft,ind] = max([(v,i) for i,v in enumerate(fdata)])
    plt.clf()
    plt.ylabel('FFT amplitude')
    plt.xlabel('Frequencies')
    plt.semilogy(fs,fdata)
    plt.savefig(fn+str(stepnum)+'.png')
    return [maxfft,fs[ind]] 

def front_back_addition(tdata):
    halfdatal = len(tdata)/2      
    frontdata = tdata[:halfdatal]
    backdata = list(reversed(tdata[halfdatal:]))[:halfdatal]
    return map(lambda f,b:f+b,frontdata,backdata)
    
def fft_comparison(data,stepnums,fn):
    plt.clf()
    plt.cla()
    plt.close('all')
    firstfdata = np.abs(fp.fft(data['wavefunction'][stepnums[0]]))
    firsttotal = front_back_addition(firstfdata)
    for stepnum in stepnums:
        fdata = np.abs(fp.fft(data['wavefunction'][stepnum]))
        totaldata = front_back_addition(fdata)
        compdata = map(lambda firstf,nextf:(1.0*nextf)/firstf,firsttotal,totaldata)
        binsize = (1.0*data['maximum_x'])/data['numxs']
        wp_energy = data['k0']**2/(2*data['mass'])
        fs = map(lambda n:(n*binsize)**2/wp_energy,range(len(totaldata)))
        plt.clf()
        plt.cla()
        plt.close('all')
        plt.ylabel('Normalised FFT amplitude')
        plt.xlabel('Energy in units of wavepacket energy')
        enmarks = range(int(fs[-1]))
        plt.xticks(enmarks)
        plt.axvline(x=1,color='r')
        plt.axhline(y=1,color='r')
        plt.semilogy(fs,compdata)
        plt.ylim( (10**(-3),10**4) )
        plt.savefig(fn+str(stepnum)+'.png')
        plt.clf()
        plt.cla()
        plt.close('all')

fft_comparison(simres,range(0,1001,100),'fftnorm')