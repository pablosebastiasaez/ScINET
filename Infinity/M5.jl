#!/sw/spack-levante/julia-1.7.0-wm6d6v/bin/julia

#SBATCH --job-name=...
#SBATCH --partition=compute
#SBATCH --account=...
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=128  
#SBATCH --time=08:00:00  
#SBATCH --output=julia-%j.out
#SBATCH --error=julia-%j.err

using Distributed
using ClusterManagers

# Add workers
addprocs_slurm(Int(128*10), nodes=10, partition="compute")
println("Number of workers: $(Distributed.nworkers())")
println("workers added")
flush(stdout)

@everywhere begin
    using ProgressMeter
    using SpecialFunctions
    using NaNStatistics
    using NPZ

    @inline function trapz(x, y, dim)
        # x: coordinate of integration
        # y: function to integrate
        # dim: dimension to integrate
        if dim == 1
            return sum((x[i+1] - x[i]) * (y[i+1] + y[i]) / 2 for i in 1:length(x)-1)
        elseif dim == 2
            return sum((x[i+1] - x[i]) * (y[:, i+1] + y[:, i]) / 2 for i in 1:length(x)-1)
        elseif dim == 3
            return sum((x[i+1] - x[i]) * (y[:,:, i+1] + y[:,:, i]) / 2 for i in 1:length(x)-1)
        else
            error("Only integration along dimensions 1 and 2 are supported.")
        end
    end

    @inline function scatter_om_m(mu, pmu, om,m,f,f2,N,N2,Nf2,ep1,ep2,lom1,n18,OM1,lx1,d05,m_end,mc,sig,ms,inte_x1_sng,inte_x1_om1_sng,energy,k_end,t1,a, c2star)
            delmu = pmu
            om02 = om^2
            c = sqrt((om02 - f2) / Nf2)
            S = sqrt((N2 - om02) / Nf2)
            Ct = c / S
            s = sig * S
            ct = c / s
            Kh2d_ = sqrt(abs((f2 - om02) * m^2 / (om02 - N2)))

            # define om1-ranges for both mu=+-  
            lom1new = 0
            if mu == -1
                if om < N - f
                    OM1 = range(om + f, stop = N * ep2, length = lom1 + n18)
                    lom1new = lom1 + n18
                else
                    lom1new = 0
                end
            else
                if om > 2.0 * f * ep1
                    OM1 = range(f * ep1, stop = om - f, length = lom1+n18)
                    lom1new = length(OM1)
                else
                    lom1new = 0
                end
            end
            
            # m loop
            mm = sig * m
            # sig1 loop
            psig1 = 0
            for sig1 in [-1, 1]
                psig1 = psig1 + 1
                
                # om1 loop
                if lom1new == 0
                    continue
                else
                    for iom1 in 1:lom1new
                        om1 = OM1[iom1]
                    
                        om2 = mu * (om - om1)
                        if om2 < f * ep1
                            continue
                        end
                        
                        om12 = om1^2
                        om22 = om2^2

                        c1 = sqrt((om12 - f2) / Nf2)
                        S1 = sqrt((N2 - om12) / Nf2)
                        Ct1 = c1 / S1
                        c22 = (om22 - f2) / Nf2
                        s22 = (N2 - om22) / Nf2
                        ct22 = c22 / s22
                        c2 = sqrt(c22)
                        S2 = sqrt(s22)
                        s1 = sig1 * S1
                        ct1 = c1 / s1
                        
                        # boundaries of y1 domain
                        C = ct22 / (ct1 * ct)
                        A = (om12 - om22) / (N2 - om22) / c1^2
                        B = (om02 - om22) / (N2 - om22) / c^2
                        aplus = real((1.0 - C) / A)
                        aminus = (-1.0 - C) / A
                        b = B / A
                        if om1 <= om / 2.0
                            y1min = aplus + sqrt(aplus^2 - b)
                            y1max = aminus + sqrt(aminus^2 - b)
                            caseABC = "I"
                        elseif om1 >= 2 * om
                            y1min = aminus + sqrt(aminus^2 - b)
                            y1max = aplus + sqrt(aplus^2 - b)
                            caseABC = "III"
                        else
                            y1min = aplus - sqrt(abs(aplus^2 - b))
                            y1max = aplus + sqrt(abs(aplus^2 - b))
                            caseABC = "II"
                        end
                        
                        y1min = real(y1min)
                        y1max = real(y1max)

                        entscheider = abs(y1max / y1min - 1.0)
                        if entscheider < 1e-6
                            continue
                        end

                        x1min = y1min * Ct / Ct1
                        x1max = y1max * Ct / Ct1

                        # Step 1: Grid setup
                        dx11 = (x1max - x1min) / lx1 / 10000.0
                        dx11 = (x1max - x1min - 2.0 * dx11) / lx1
                        x1 = range(x1min + dx11, stop = x1max - dx11, length = lx1)

                        if x1min + dx11 > x1max - dx11
                            error("x1 array wrong")
                        end

                        intemu_ = ones(length(x1))
                        @inbounds @simd for ix1 in 1:lx1
                            # Step 2: Derived arrays
                            xx1 = x1[ix1]
                            m1 =  xx1 * m
                            mm1 =  sig1 * m1
                            y1 =  xx1 * Ct1 / Ct
                            
                            mm2 =  mu * (mm - mm1)
                            m2 = abs(mm2)
                            x2 =  m2 / m
                            sig2 =  sign(mm2)
                            s2 =  sig2 * sqrt(s22)

                            # Step 3: Geometry
                            cgam = real((1.0 + y1^2 - ct22 * (1.0 / Ct - sig * sig1 .* y1 / Ct1)^2) / (2.0 * y1))
                            cgam = ifelse(abs(cgam) >= 1.0, NaN, cgam)                            
                            sgam =  sqrt(1.0 - cgam^2)
                            y2 =  sqrt(1.0 + y1^2 - 2.0 * y1 * cgam)
                            
                            # Step 4: Jacobians
                            J0m2 =  Nf2 * om / (N2 - om02)^2
                            J1m2 =  xx1^2 * Nf2 * om1 / (N2 - om12)^2
                            J2m2 =  x2^2 * Nf2 * om2 / (N2 - om22)^2

                            # Step 5: Cross-section terms
                            immag = Complex(0, 1)
                            G1 =  (cgam - y1 + immag * mu * f / om2 * sgam) * s2 * c1 * mu - y2 * c2 * s1
                            G2 =  (cgam - immag * f / om1 * sgam) * s1 * c - c1 * s
                            G3 =  (1.0 - y1 * cgam + immag * mu * f / om2 * sgam * y1) * s2 * c * mu - y2 * c2 * s
                            G4 =  (-cgam + immag * f / om * sgam) * s * c1 + c * s1

                            V =  c^2 * s * y1 * G1 * G2 + c1^2 * s1 * G3 * G4 - mu * c2^2 * s2 * y1 * G2 * G4
                            VV =  V * conj(V)
                            Tk2 =  (π / 8) * Nf2^2 / (om * om1 * om2) * VV / (y2^2 * c1^2 * c^2)
                            tlTmu =  Tk2 / (y1 * sgam)

                            # Step 6: Compute GM-spectrum
                            KK = sqrt(m^2 * (f2 - om02)/(om02 - N2))
                            Ksqr = KK^2 + m^2
                            ms = sqrt((N2 - om02) / c2star)
                            Atl = (1.0 + (m / ms)^a)^(-t1 / a) 
                            GM0 =  energy * Atl  / ms / ((N2 * KK^2 + f2 * m^2)*sqrt(Ksqr)) / (KK) *J0m2 * m^4

                            KK = sqrt(m1^2 * (f2 - om12)/(om12 - N2))
                            Ksqr = KK^2 + m1^2
                            ms = sqrt((N2 - om12) / c2star)
                            Atl = (1.0 + (m1 / ms)^a)^(-t1 / a) 
                            GM1 =  energy * Atl  / ms  / ((N2 * KK^2 + f2 * m1^2)*sqrt(Ksqr)) / (KK) *J1m2 * m1^2 * m^2

                            KK = sqrt(m2^2 * (f2 - om22)/(om22 - N2))
                            Ksqr = KK^2 + m2^2
                            ms = sqrt((N2 - om22) / c2star)
                            Atl = (1.0 + (m2 / ms)^a)^(-t1 / a) 
                            GM2 =  energy * Atl / ms / ((N2 * KK^2 + f2 * m2^2)*sqrt(Ksqr)) / (KK) *J2m2  * m2^2 *m^2

                            # Step 7: Convert GM to action
                            GM0 =  GM0 / om
                            GM1 =  GM1 / om1
                            GM2 =  GM2 / om2

                            # Step 8: Compute integrand
                            intemu =  delmu * tlTmu * (J0m2 * GM1 * GM2 - mu * J2m2 * GM1 * GM0 - J1m2 * GM2 * GM0)
                            outl1 =  (1e-3) <= m1 <= m_end
                            outl2 =  (1e-3) <= m2 <= m_end
                            outl3 =  (1e-4) <= y1 * Kh2d_ <= k_end
                            outl4 =  (1e-4) <= y2 * Kh2d_ <= k_end
                            outl5 =  (1e-4) <= Kh2d_ <= k_end
                            intemu_[ix1] =  intemu * outl1 * outl2 * outl3 * outl4 * outl5
    
                        end
                        # integrate over x1, sng
                        inteJ = @. intemu_ * sqrt(x1 - x1min) * sqrt(x1max - x1)
                        gg1 = inteJ[1]
                        ggn = inteJ[end]
                        Itilde = @. intemu_ - (gg1 / sqrt(x1max - x1min)) / sqrt(x1 - x1min) - (ggn / sqrt(x1max - x1min)) / sqrt(x1max - x1)
                        Itilde_x1 = trapz(x1, Itilde, 1)#[1]
                        inte_x1_sng[iom1] = Itilde_x1 + 2.0 * (gg1 + ggn)
                    end
                end

                inte_x1_sng = @. ifelse(isnan(inte_x1_sng), 0, inte_x1_sng)

                inte_x1_om1_sng[psig1] = trapz(OM1, inte_x1_sng, 1)#[1]
            end
            # sum over sig1
            inte_x1_om1_sig1_sng = nansum(inte_x1_om1_sng)
            SNG = m^3 * om * inte_x1_om1_sig1_sng
        return(SNG)
    end

    function scatter_func(om,m,f,f2,N,N2,Nf2,ep1,ep2,lom1,n18,OM1,lx1,d05,m_end,mc,sig,ms,energy,k_end,t1,a, c2star)
        SNG = zeros(Float64,  2)
        mu = 1
        pmu = 1.0
        inte_x1_sng = zeros(Float64, lom1 + n18)
        inte_x1_om1_sng = zeros(Float64, 2)

        SNG[1] = scatter_om_m(mu, pmu, om,m,f,f2,N,N2,Nf2,ep1,ep2,lom1,n18,OM1,lx1,d05,m_end,mc,sig,ms,inte_x1_sng,inte_x1_om1_sng,energy,k_end,t1,a, c2star)
        mu = -1
        pmu = 2.0
        inte_x1_sng = zeros(Float64, lom1 + n18)
        inte_x1_om1_sng = zeros(Float64, 2)
        
        SNG[2] = scatter_om_m(mu, pmu, om,m,f,f2,N,N2,Nf2,ep1,ep2,lom1,n18,OM1,lx1,d05,m_end,mc,sig,ms,inte_x1_sng,inte_x1_om1_sng,energy,k_end,t1,a, c2star)
        return(om,m,SNG)  
    end
end



# Parameters
ep1 = 1 + 0.001  # to start om at frq higher than f
ep2 = 1 - 0.001  # to stop om at frq less than N
f = 10e-5
N100 = 50
N = N100 * f
mstar = 0.01
mc = 100 * mstar
mlow = 0.1 * mstar
ms = mstar
sig = 1

m_start = 1e-3
m_end = 1   # [1/m]
k_start= 1e-4
k_end = m_end/10

Ksqr = k_start^2 + m_end^2
omeg = ((N^2 * k_start^2 + f^2 * m_end^2) / Ksqr)^0.5
ep1 = omeg/f

Ksqr = k_end^2 + m_start^2
omeg = ((N^2 * k_end^2 + f^2 * m_start^2) / Ksqr)^0.5
ep2 = omeg/N

# Derived values
f2 = f^2
N2 = N^2
Nf2 = N2 - f2

# Set om1 m1 grid
lom1 = 1862
n18 = 0#128#500
d05 = 0.9999  # dim omega1, n18 must be even
lx1 = 1860  # for integration array x1 = m1/m

# Set omega-m grid
lom = lom1  # dim omega
lm = lx1  # dim m

# New m grid
M = 10 .^ range(log10(m_start), stop=log10(m_end), length=lm)
Mend = M[end]
Mstart = M[1]
dMi = diff(M)
dM = vcat(dMi, dMi[end])

# New omega array
OM = 10 .^ range(log10(f * ep1), log10(N * ep2), lom)
OM1 = zeros(Float64, lom1 + n18)

mesh_om = zeros(Float64,lm, lom)
mesh_m = zeros(Float64,lm, lom)

for im in 1:lm
    for iom in 1:lom
        mesh_om[im, iom] = OM[iom]
        mesh_m[im, iom] = M[im]
    end
end
my_om = vec(mesh_om)
my_m = vec(mesh_m)

l_om_m = Int(lom*lm)

# Precompute parts of GM
t1 = 2
a = 2
E0 = 3e-3
nb = 2 / π
na = a * gamma(t1 / a) / gamma(1 / a) / gamma((t1 - 1) / a)
m_star = 0.01
m_star2 = 0.1
c2star = N2 / m_star^2
energy = E0 * nb * na * f * sqrt(Nf2)/(4.0 * π)
KK = @. sqrt(mesh_m^2 * (f2 - mesh_om^2) / (mesh_om^2 - N2))
Ksqr = @. KK^2 + mesh_m^2
ms = @. sqrt((N2 - mesh_om^2) / c2star)
Atl = @. (1.0 + (mesh_m / ms)^a)^(-t1 / a) 
Jac = @. mesh_m^2 * mesh_om * (Nf2) / (N2 - mesh_om^2)^2
GM0 = @. energy * Atl / ms / ((N2 * KK^2 + f2 * mesh_m^2)*sqrt(Ksqr)) / KK *Jac * mesh_m^2

collected_results = @showprogress @distributed (vcat) for i in 1:l_om_m
    scatter_func(my_om[i], my_m[i], f, f2, N, N2, Nf2, ep1, ep2, lom1, n18, OM1, lx1, d05, m_end, mstar, sig, ms,energy,k_end,t1,a, c2star)
end

# Separate the collected results into individual arrays
om  = [df[1] for df in collected_results]
kz  = [df[2] for df in collected_results]
SNG = [df[3] for df in collected_results]
SNG_p = [SNG[i][1] for i in 1:length(SNG)]
SNG_m = [SNG[i][2] for i in 1:length(SNG)]

om_ = reshape(om, size(M)[1],size(OM)[1])
kz_ = reshape(kz, size(M)[1],size(OM)[1])
sng_p_ = reshape(SNG_p, size(M)[1],size(OM)[1])
sng_m_ = reshape(SNG_m, size(M)[1],size(OM)[1])

println("saving model")
flush(stdout)
name = "M5.npz"
npzwrite(name, Dict(
    "om" => om_,
    "kz" => kz_,
    "SNGp" => sng_p_,
    "SNGm" => sng_m_,
    "GM" => GM0,
))
println("Model saved")
flush(stdout)
