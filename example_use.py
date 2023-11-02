from CahnHilliard import *

L = 40 # side length of domain
N = 400 # number of points per spatial dimension
dx = L / N # spatial step along x
dy = L / N # spatial step along y

x, y = np.meshgrid(np.linspace(-L/2, L/2, N), np.linspace(-L/2, L/2, N))
mean = 0.0 # average concentration
conc0 = np.zeros((N, N)) + mean

D = dx**2 * 3 # diffusivity cm**2 / s
gamma = 500
beta = N / dx / gamma # in this way lmax is 2*pi * L/gamma
a = dx**4 * beta**2 # fatstest growing wavevector is 1/sqrt(a)

amp = 0.01
conc0 = conc0 + amp * np.random.randn(N, N) # initial concentration

print(f'Average concentration: {np.mean(conc0):2f}\n')

tmax = 4.5
Nt = 500
time = np.linspace(0, tmax, Nt)

CH_system = Sol_CahnHilliard(L, N, D, a)
# Simulate the time evolution of the concentration field
CH_system.compute_sol(conc0, time)
# Set the step size for data analysis and visualization
CH_system.set_step(10)

# Compute the average Fourier transform of concentration profiles
CH_system.ComputeFT()
# Compute concentration histograms over time
CH_system.Compute_histo()
# Create GIF animations
CH_system.MakeGif_sol(file_name='cahn_hilliard.gif')
CH_system.MakeGif_FT(filename='ft_vs_time.gif')
CH_system.MakeGif_tot(file_name='cahn_hilliard_tot.gif')