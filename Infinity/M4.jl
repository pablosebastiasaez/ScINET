#!/sw/spack-levante/julia-1.7.0-wm6d6v/bin/julia

#SBATCH --job-name=...
#SBATCH --partition=compute
#SBATCH --account=...
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=128  # Request all CPUs on the node
#SBATCH --time=08:00:00  # Adjust based on your expected runtime
#SBATCH --output=julia-%j.out
#SBATCH --error=julia-%j.err

using Distributed
using ClusterManagers

# Add workers
addprocs_slurm(Int(5*128), nodes=5, partition="compute")
println("Number of workers: $(Distributed.nworkers())")
println("workers added")
flush(stdout)

@everywhere begin
    using ProgressMeter
    using SpecialFunctions
    using LinearAlgebra
    using NaNStatistics
    using Roots
    using NPZ

    @inline function gmu_reson(x,k1, k2, m, om, N2, f2, mu)
        # Resonance condition gmu = 0
        kx = sqrt.(k1.^2 .+ x.^2)
        om1 = sqrt.(N2 .* k1.^2 .+ f2 .* x.^2) ./ kx
        m2 = mu .* (m .- x)
        km2 = sqrt.(k2.^2 .+ m2.^2)
        om2 = sqrt.(N2 .* k2.^2 .+ f2 .* m2.^2) ./ km2
        g = om .- om1 .- mu.*om2
        return g
    end
    
    @inline function spectr(kx, mx , N2, f2, slope, ms,eps_h,eps_v)
        kkx = abs.(kx) .+ eps_h
        mkx = abs.(mx) .+ eps_v
        t = slope
        s = 2
        E0 = 3e-3
        nb = 2 ./ π
        m_star = 0.01
        na = s .* gamma(t ./ s) ./ gamma(1 ./ s) ./ gamma((t .- 1) ./ s)
        fac = E0 .* nb .* na .* sqrt.(f2) .* sqrt.(N2 .- f2)
        KK = kkx
        Ksqr = KK.^2 .+ mkx.^2
        om2 = (N2 .* KK.^2 .+ f2 .* mkx.^2) ./ Ksqr
        c2star = N2 ./ m_star.^2
        ms = sqrt.((N2 .- om2) ./ c2star)
        Atl = (1.0 .+ abs.(mkx ./ ms).^s).^(.-t ./ s)
        energy = fac .* Atl .* mkx.^2 ./ (4.0 .* π .* ms)
        GM1 = ifelse.(kx.^2 .+ mx.^2 .== 0 , 0.0 , energy ./ ((N2 .* KK.^2 .+ f2 .* mkx.^2).*sqrt.(Ksqr))./ (KK) .*KK)
        GM2 = ifelse.(KK .> 0.1, 0.0, GM1)
        GM = ifelse.(mkx.^2 .> 1, 0.0, GM2)
        return GM
    end

    # Define the trapz function
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

    function root_find(xxx, k1_i, k2_i, m, om, N2, f2, mu)
        # Minimum finding algorithm - Find roots for each element in k1_i, k2_i
        root_sum_p(k1_i, k2_i, m, om, N2, f2, mu) = find_zero(x -> gmu_reson.(x, k1_i, k2_i, m, om, N2, f2, mu), xxx)
        kz1_b = root_sum_p.( k1_i, k2_i, m, om, N2, f2, mu)
        return(kz1_b)
    end
    
    function dirks_scatter(K,lk,M,lm,PHI1,yPHI1,lphi1,K1,xK1,lk1,N,N2,f,f2,Nf2,sig,mu_,ms,slope,delmu_,x0,x1,m_start,k_start,m_end,k_end,intemu1,eps_h,eps_v)
        k = K
        m_ = M
        phi = 0.0 # (no loop necessary)
        KK = sqrt.(k.^2 .+ m_.^2)
        om_ = sqrt.(N2 .* k.^2 .+ f2 .* m_.^2) ./ KK
        c = k ./ KK
        s = sig .* m_ ./ KK

        k1_ = xK1
        y1_ = k1_ ./ k
        phi1 = yPHI1
        cgam_ = cos.(phi .- phi1)
        sgam_ = sqrt.(1 .- cgam_.^2)  # abs value for sgam
        k2_ = sqrt.(k.^2 .+ k1_.^2 .- 2 .* k .* k1_ .* cgam_)
        y2_ = k2_ ./ k

        # Sum interactions
        mu = mu_[1]
        delmu = delmu_[1]
        # find m1
        result = zeros(lphi1,lk1,2)
        for psig1 in 1:2  # sign of m1
            if psig1 == 1  # search positive m1
                g0 = gmu_reson.(x0[1],k1_, k2_, m_, om_, N2, f2, mu) 
                g1 = gmu_reson.(x0[2],k1_, k2_, m_, om_, N2, f2, mu)
                g01 = g0 .* g1 .> 0
                xxx=x0
            elseif psig1 == 2  # search negative m1
                g0 = gmu_reson.(x1[1],k1_, k2_, m_, om_, N2, f2, mu)
                g1 = gmu_reson.(x1[2],k1_, k2_, m_, om_, N2, f2, mu)
                g01 = g0 .* g1 .> 0
                xxx= x1
            end
            
            k1_i = k1_[g01.==0]
            y1_i = y1_[g01.==0]
            cgam_i = cgam_[g01.==0]
            sgam_i = sgam_[g01.==0]
            k2_i = k2_[g01.==0]
            y2_i = y2_[g01.==0]
            indices2 = findall(g01.==0)

            if isempty(y1_i)
                step1 = zeros(Float64,  lphi1,lk1)
            else
                m1_i = root_find(xxx, k1_i, k2_i, m_, om_, N2, f2, mu)  
                sig1 = sign.(m1_i)
                m2_i = mu .* (m_ .- m1_i)  # m2 with sign
                n = ((k2_i .<= k_end) .&& (k2_i .>= k_start) .&&
                    ((abs.(m2_i) .<= m_end) .&& (abs.(m2_i) .>= m_start)) .&&
                    ((abs.(m1_i) .<= m_end) .&& abs.(m1_i) .>= m_start))
                
                k1 = k1_i[n]
                y1 = y1_i[n]
                cgam = cgam_i[n]
                sgam = sgam_i[n]
                k2 = k2_i[n]
                y2 = y2_i[n]
                m1 = m1_i[n]
                m2 = m2_i[n]
                
                m = (m_ .* ones(Float64, length(m1)))
                om = (om_ .* ones(Float64, length(m1)))

                sig2 = sign.(m2)
                KK1 = sqrt.(k1.^2 .+ m1.^2)
                KK2 = sqrt.(k2.^2 .+ m2.^2)

                om1 = sqrt.(N2 .* k1.^2 .+ f2 .* m1.^2) ./ KK1
                om12 = om1.^2
                om2 = mu .* (om .- om1)
                om22 = om2.^2

                c1 = k1 ./ KK1
                s1 = m1 ./ KK1
                c2 = k2 ./ KK2
                s2 = m2 ./ KK2

                # cross section Tmu
                G1 = (cgam .- y1 .+ Complex(0, 1) .* mu .* f ./ om2 .* sgam) .* s2 .* c1 .* mu .- y2 .* c2 .* s1
                G2 = (cgam .- Complex(0, 1) .* f ./ om1 .* sgam) .* s1 .* c .- c1 .* s
                G3 = (1.0 .- y1 .* cgam .+ Complex(0, 1) .* mu .* f ./ om2 .* sgam .* y1) .* s2 .* c .* mu .- y2 .* c2 .* s
                G4 = (.-cgam .+ Complex(0, 1) .* f ./ om .* sgam) .* s .* c1 .+ c .* s1
                V = c.^2 .* s .* y1 .* G1 .* G2 .+ c1.^2 .* s1 .* G3 .* G4 .- mu .* c2.^2 .* s2 .* y1 .* G2 .* G4
                VV = abs.(V).^2
                Tmu = k.^2 .* ((pi ./ 8.0) .* Nf2.^2 ./ (om .* om1 .* om2)) .* VV ./ (y2.^2 .* c1.^2 .* c.^2)  # now unscaled
                gs = m1 ./ om1 .* (om12 .- f2) ./ (k1.^2 .+ m1.^2) .- m2 ./ om2 .* (om22 .- f2) ./ (k2.^2 .+ m2.^2)
                gs = abs.(gs)
                tlTmu = Tmu ./ (k2 .* gs)
                
                # Compute the GM spectrum as wave action
                GM0 = spectr.(k, m, N2, f2, slope, ms,eps_h,eps_v) ./ om
                GM1 = spectr.(k1, m1, N2, f2, slope, ms,eps_h,eps_v) ./om1
                GM2 = spectr.(k2, m2, N2, f2, slope, ms,eps_h,eps_v) ./om2

                isn1 = delmu .* tlTmu .* (k .* GM1 .* GM2 .- mu .* k2 .* GM1 .* GM0 .- k1 .* GM2 .* GM0)
                isn = ifelse.(isnan.(isn1), 0.0, isn1)
                indices = findall(n)
                step = zeros(Float64, length(k1_i))
                step[indices] = isn
                step01 = zeros(Float64, length(k1_))
                step01[indices2] = step
                step1 = reshape(step01, lphi1,lk1)
            end
            result[1:end,1:end,psig1] = step1
            intemu1 = nansum(result, dims=3)[:,:,1]
        end  # sig1
        dtE_sum = trapz(K1, trapz(PHI1, transpose(intemu1), 2), 1)[1] .* om_ .* 2

        # Difference interactions
        k = K
        m_ = M
        phi = 0.0  # (no loop necessary)
        KK = sqrt.(k.^2 .+ m_.^2)
        om_ = sqrt.(N2 .* k.^2 .+ f2 .* m_.^2) ./ KK
        c = k ./ KK
        s = sig .* m_ ./ KK

        k1_ = xK1
        y1_ = k1_ ./ k
        phi1 = yPHI1
        cgam_ = cos.(phi .- phi1)
        sgam_ = sqrt.(1 .- cgam_.^2)  # abs value for sgam
        k2_ = sqrt.(k.^2 .+ k1_.^2 .- 2 .* k .* k1_ .* cgam_)
        y2_ = k2_ ./ k
        mu = mu_[2]
        delmu = delmu_[2]
            
        # find m1
        result = zeros(lphi1,lk1,2)
        for psig1 in 1:2  # sign of m1
            if psig1 == 1  # search positive m1
                g0 = gmu_reson.(x0[1],k1_, k2_, m_, om_, N2, f2, mu) 
                g1 = gmu_reson.(x0[2],k1_, k2_, m_, om_, N2, f2, mu)
                g01 = g0 .* g1 .> 0
                xxx=x0
            elseif psig1 == 2  # search negative m1
                g0 = gmu_reson.(x1[1],k1_, k2_, m_, om_, N2, f2, mu)
                g1 = gmu_reson.(x1[2],k1_, k2_, m_, om_, N2, f2, mu)
                g01 = g0 .* g1 .> 0
                xxx= x1
            end
            
            k1_i = k1_[g01.==0]
            y1_i = y1_[g01.==0]
            cgam_i = cgam_[g01.==0]
            sgam_i = sgam_[g01.==0]
            k2_i = k2_[g01.==0]
            y2_i = y2_[g01.==0]
            indices2 = findall(g01.==0)

            if isempty(y1_i)
                step1 = zeros(Float64,  lphi1,lk1)
            else
                m1_i = root_find(xxx, k1_i, k2_i, m_, om_, N2, f2, mu)  
                sig1 = sign.(m1_i)
                m2_i = mu .* (m_ .- m1_i)  # m2 with sign

                n = ((k2_i .<= k_end) .&& (k2_i .>= k_start) .&&
                    ((abs.(m2_i) .<= m_end) .&& (abs.(m2_i) .>= m_start)) .&&
                    ((abs.(m1_i) .<= m_end) .&& abs.(m1_i) .>= m_start))
                
                k1 = k1_i[n]
                y1 = y1_i[n]
                cgam = cgam_i[n]
                sgam = sgam_i[n]
                k2 = k2_i[n]
                y2 = y2_i[n]
                m1 = m1_i[n]
                m2 = m2_i[n]
                
                m = (m_ .* ones(Float64, length(m1)))
                om = (om_ .* ones(Float64, length(m1)))

                sig2 = sign.(m2)
                KK1 = sqrt.(k1.^2 .+ m1.^2)
                KK2 = sqrt.(k2.^2 .+ m2.^2)

                om1 = sqrt.(N2 .* k1.^2 .+ f2 .* m1.^2) ./ KK1
                om12 = om1.^2
                om2 = mu .* (om .- om1)
                om22 = om2.^2

                c1 = k1 ./ KK1
                s1 = m1 ./ KK1
                c2 = k2 ./ KK2
                s2 = m2 ./ KK2

                # cross section Tmu
                G1 = (cgam .- y1 .+ Complex(0, 1) .* mu .* f ./ om2 .* sgam) .* s2 .* c1 .* mu .- y2 .* c2 .* s1
                G2 = (cgam .- Complex(0, 1) .* f ./ om1 .* sgam) .* s1 .* c .- c1 .* s
                G3 = (1.0 .- y1 .* cgam .+ Complex(0, 1) .* mu .* f ./ om2 .* sgam .* y1) .* s2 .* c .* mu .- y2 .* c2 .* s
                G4 = (.-cgam .+ Complex(0, 1) .* f ./ om .* sgam) .* s .* c1 .+ c .* s1
                V = c.^2 .* s .* y1 .* G1 .* G2 .+ c1.^2 .* s1 .* G3 .* G4 .- mu .* c2.^2 .* s2 .* y1 .* G2 .* G4
                VV = abs.(V).^2
                Tmu = k.^2 .* ((pi ./ 8.0) .* Nf2.^2 ./ (om .* om1 .* om2)) .* VV ./ (y2.^2 .* c1.^2 .* c.^2)  # now unscaled
                gs = m1 ./ om1 .* (om12 .- f2) ./ (k1.^2 .+ m1.^2) .- m2 ./ om2 .* (om22 .- f2) ./ (k2.^2 .+ m2.^2)
                gs = abs.(gs)
                tlTmu = Tmu ./ (k2 .* gs)
                
                # Compute the GM spectrum as wave action
                GM0 = spectr.(k, m, N2, f2, slope, ms,eps_h,eps_v) ./ om
                GM1 = spectr.(k1, m1, N2, f2, slope, ms,eps_h,eps_v) ./om1
                GM2 = spectr.(k2, m2, N2, f2, slope, ms,eps_h,eps_v) ./om2

                isn1 = delmu .* tlTmu .* (k .* GM1 .* GM2 .- mu .* k2 .* GM1 .* GM0 .- k1 .* GM2 .* GM0)
                isn = ifelse.(isnan.(isn1), 0.0, isn1)
                indices = findall(n)
                step = zeros(Float64, length(k1_i))
                step[indices] = isn
                step01 = zeros(Float64, length(k1_))
                step01[indices2] = step
                step1 = reshape(step01, lphi1,lk1)
            end
            result[1:end,1:end,psig1] = step1
            intemu1 = nansum(result, dims=3)[:,:,1]
        end  # sig1
        dtE_diff = trapz(K1, trapz(PHI1, transpose(intemu1), 2), 1)[1] .* om_ .* 2
    return(k,m_,dtE_sum,dtE_diff)
    end
end

# Define Parameters
f = 1e-4
N = 50*f
f2=f^2
N2=N^2
Nf2=N2-f2

ms=0.01
sig=1
mc=100*ms
slope=2.0
cstar=N/ms
mu = [1,-1]
delmu = [1,2]

# Set up grids
lk=1060
lm=1040
lk1=lk
lphi1=1020
#m grid
m_start= 1e-3      
m_end=   1.0    
M = 10 .^ range(log10(m_start), stop=log10(m_end), length=lm)
Mend=M[end]
Mstart=M[1]
#k grid
k_start= 1e-4    
k_end=  0.1
K = 10 .^ range(log10(k_start), stop=log10(k_end), length=lk)
Kend=K[end]
Kstart=K[1]
K1=K
#%phi1 grid
phi1_start=0.00001
phi1_end=pi # only half needed - isotropy -> factor 2 at end
PHI1 = range(phi1_start, phi1_end, length=lphi1)
dphi1 = abs(PHI1[2] - PHI1[1])

eps_h = k_start*0.5
eps_v = m_start*0.5
# allocations
intemu1=zeros(lphi1,lk1) # sum over sig1

## zero search
mstart=m_start
mend=m_end
x0 = [mstart, mend] # initial point for m1 positive
x1 = [-mend, -mstart] # initial point for m1 negative

Xk1,Yphi1 = zeros(Float64, lphi1,lk1),  zeros(Float64, lphi1,lk1)
for j in 1:lphi1
    for i in 1:lk1
        Xk1[j,i]   = K1[i]
        Yphi1[j,i] = PHI1[j]
    end
end
xK1,yPHI1 = vec(Xk1), vec(Yphi1)


Z2d,X2d = zeros(Float64, lm,lk),  zeros(Float64, lm,lk)
for i in 1:lm
    for j in 1:lk
        X2d[i,j] = K[j]
        Z2d[i,j] = M[i]
    end
end
X2d,Z2d = vec(X2d), vec(Z2d)
N_len = Int(lm*lk)

# Initialize an empty array to collect results
collected_results = @showprogress @distributed (vcat) for i in 1:N_len
    #println(i , " from ", N)
    dirks_scatter(X2d[i],lk,Z2d[i],lm,PHI1,yPHI1,lphi1,K1,xK1,lk1,N,N2,f,f2,Nf2,sig,mu,ms,slope,delmu,x0,x1,m_start,k_start,m_end,k_end,intemu1,eps_h,eps_v)
end

# Separate the collected results into individual arrays
kh = [df[1] for df in collected_results]
kz = [df[2] for df in collected_results]
dt_sum = [df[3] for df in collected_results]
dt_diff = [df[4] for df in collected_results]

println("saving model")
flush(stdout)
name = "M4.npz"
npzwrite(name, Dict(
    "kx" => kh,
    "kz" => kz,
    "ww_sum" => dt_sum,
    "ww_diff" => dt_diff,
))
println("Model saved")
flush(stdout)
