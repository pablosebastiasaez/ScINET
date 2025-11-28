import numpy as np
cimport numpy as cnp
from scipy.special import gamma

# omega2_true function
def omega2_true(cnp.ndarray[double, ndim=1] kx, cnp.ndarray[double, ndim=1] ky, cnp.ndarray[double, ndim=1] kz, double s, double f0, double N0):
    cdef int n = kx.shape[0]
    cdef cnp.ndarray[double, ndim=1] om2 = np.zeros(n, dtype=np.float64)
    cdef double ksqr, k2
    for i in range(n):
        ksqr = kx[i]**2 + ky[i]**2
        k2 = ksqr + kz[i]**2
        if k2 > 0:
            om2[i] = (N0**2 * ksqr + f0**2 * kz[i]**2) / k2
        else:
            om2[i] = 0.0
    return om2

# coeff_p function
def coeff_p(cnp.ndarray[double, ndim=1] kx, cnp.ndarray[double, ndim=1] ky, cnp.ndarray[double, ndim=1] kz, 
            cnp.ndarray[double, ndim=1] om, double s, double f0, double N0):
    cdef int n = kx.shape[0]
    cdef cnp.ndarray[double, ndim=1] p = np.zeros(n, dtype=np.float64)
    cdef double ksqr, k2, om2
    for i in range(n):
        ksqr = kx[i]**2 + ky[i]**2
        k2 = ksqr + kz[i]**2
        om2 = om[i]**2
        if s == 0:
            p[i] = abs(N0**2 * f0**2) / (N0**2 * ksqr + f0**2 * kz[i]**2)
        else:
            if k2 == 0:
                p[i] = np.nan
            else:
                p[i] = abs((N0**2 - s**2 * om2) * (s**2 * om2 - f0**2)) / ((1 + s**2) * k2 * om2)
            if ksqr == 0 and kz[i]**2 > 0:
                p[i] = 0.5
            if ksqr > 0 and kz[i] == 0:
                p[i] = N0**2 / 2.0
    return p

# vec_q function
def vec_q(cnp.ndarray[double, ndim=1] kx, cnp.ndarray[double, ndim=1] ky, cnp.ndarray[double, ndim=1] kz, 
          cnp.ndarray[double, ndim=1] om, double s, double f0, double N0):
    cdef int n = kx.shape[0]
    cdef cnp.ndarray[complex, ndim=1] qx = np.zeros(n, dtype=np.complex128)
    cdef cnp.ndarray[complex, ndim=1] qy = np.zeros(n, dtype=np.complex128)
    cdef cnp.ndarray[complex, ndim=1] qz = np.zeros(n, dtype=np.complex128)
    cdef cnp.ndarray[complex, ndim=1] qb = np.zeros(n, dtype=np.complex128)
    cdef double ksqr, om2, division1, division2
    cdef complex icpmx = 1j
    for i in range(n):
        ksqr = kx[i]**2 + ky[i]**2
        om2 = om[i]**2
        if s == 0:
            division1 = f0**2
            division2 = N0**2
            qx[i] = (-icpmx * f0 * ky[i]) / division1
            qy[i] = (+icpmx * f0 * kx[i]) / division1
            qz[i] = 0
            qb[i] = icpmx * kz[i] * N0**2 / division2
        else:
            division1 = f0**2 - s**2 * om2 if ksqr != 0 else 1
            division2 = N0**2 - s**2 * om2 if kz[i] != 0 else 1

            if ksqr == 0 and kz[i]**2 > 0: 
                qx[i] = -icpmx*s
                qy[i] = 1.0
                qz[i] = 0.0
                qb[i] = 0.0
            if ksqr > 0 and kz[i] == 0:
                qx[i] = 0.0
                qy[i] = 0.0
                qz[i] = -icpmx*s/N0
                qb[i] = 1.0
            if ksqr > 0 and kz[i]**2 > 0:
                qx[i] = (-icpmx * f0 * ky[i] + s * om[i] * kx[i]) / division1
                qy[i] = (+icpmx * f0 * kx[i] + s * om[i] * ky[i]) / division1
                qz[i] = kz[i] * s * om[i] / division2
                qb[i] = icpmx * kz[i] * N0**2 / division2
    return qx, qy, qz, qb

# vec_p function
def vec_p(cnp.ndarray[double, ndim=1] kx, cnp.ndarray[double, ndim=1] ky, cnp.ndarray[double, ndim=1] kz, 
          cnp.ndarray[double, ndim=1] om, double s, double f0, double N0):
    cdef int n = kx.shape[0]
    cdef cnp.ndarray[complex, ndim=1] px = np.zeros(n, dtype=np.complex128)
    cdef cnp.ndarray[complex, ndim=1] py = np.zeros(n, dtype=np.complex128)
    cdef cnp.ndarray[complex, ndim=1] pz = np.zeros(n, dtype=np.complex128)
    cdef cnp.ndarray[complex, ndim=1] pb = np.zeros(n, dtype=np.complex128)
    cdef cnp.ndarray[complex, ndim=1] qx, qy, qz, qb
    cdef cnp.ndarray[double, ndim=1] p0
    qx, qy, qz, qb = vec_q(kx, ky, kz, om, s, f0, N0)
    p = coeff_p(kx, ky, kz, om, s, f0, N0)
    for i in range(n):
        px[i] = p[i]* qx[i].conjugate()
        py[i] = p[i]* qy[i].conjugate()
        pz[i] = p[i]* qz[i].conjugate()
        pb[i] = p[i]* qb[i].conjugate()/(N0**2)
    return px, py, pz, pb

# coeff_C function
def coeff_C(cnp.ndarray[double, ndim=1] kx0, cnp.ndarray[double, ndim=1] ky0, cnp.ndarray[double, ndim=1] kz0, 
            cnp.ndarray[double, ndim=1] om0, double s0, 
            cnp.ndarray[double, ndim=1] kx1, cnp.ndarray[double, ndim=1] ky1, cnp.ndarray[double, ndim=1] kz1, 
            cnp.ndarray[double, ndim=1] om1, double s1, 
            cnp.ndarray[double, ndim=1] kx2, cnp.ndarray[double, ndim=1] ky2, cnp.ndarray[double, ndim=1] kz2, 
            cnp.ndarray[double, ndim=1] om2, double s2, double f0, double N0):
    cdef int n = kx0.shape[0]
    cdef cnp.ndarray[complex, ndim=1] C = np.zeros(n, dtype=np.complex128)
    cdef cnp.ndarray[complex, ndim=1] qx1, qy1, qz1, qb1
    cdef cnp.ndarray[complex, ndim=1] qx2, qy2, qz2, qb2
    cdef cnp.ndarray[complex, ndim=1] px0, py0, pz0, pb0
    cdef cnp.ndarray[double, ndim=1] p0, p1, p2
    cdef complex q1k2, q2k1  # Declare as complex to handle complex values
    px0, py0, pz0, pb0 = vec_p(kx0, ky0, kz0, om0, s0, f0, N0)
    qx1, qy1, qz1, qb1 = vec_q(kx1, ky1, kz1, om1, s1, f0, N0)
    qx2, qy2, qz2, qb2 = vec_q(kx2, ky2, kz2, om2, s2, f0, N0)
    p0 = coeff_p(kx0, ky0, kz0, om0, s0, f0, N0)
    p1 = coeff_p(kx1, ky1, kz1, om1, s1, f0, N0)
    p2 = coeff_p(kx2, ky2, kz2, om2, s2, f0, N0)
    for i in range(n):
        q1k2 = qx1[i] * kx2[i] + qy1[i] * ky2[i] + qz1[i] * kz2[i]
        q2k1 = qx2[i] * kx1[i] + qy2[i] * ky1[i] + qz2[i] * kz1[i]
        px0_qx2 = px0[i]* qx2[i]
        py0_qy2 = py0[i]* qy2[i]
        pz0_qz2 = pz0[i]* qz2[i]
        pb0_qb2 = pb0[i]* qb2[i]
        px0_qx1 = px0[i]* qx1[i]
        py0_qy1 = py0[i]* qy1[i]
        pz0_qz1 = pz0[i]* qz1[i]
        pb0_qb1 = pb0[i]* qb1[i]
        C[i] = 0.5 * np.sqrt(p0[i] * p1[i] * p2[i]) / p0[i] *((px0_qx2 + py0_qy2 + pz0_qz2 + pb0_qb2) * q1k2 +  (px0_qx1 + py0_qy1 + pz0_qz1 + pb0_qb1) * q2k1)
    return C

# Ep function
def Ep(cnp.ndarray[double, ndim=1] kx, cnp.ndarray[double, ndim=1] ky, cnp.ndarray[double, ndim=1] kz, 
       cnp.ndarray[double, ndim=1] om, double ss, double f0, double N0, double eps_h, double eps_v, double xmax, double zmax):
    cdef int n = kx.shape[0]
    cdef cnp.ndarray[double, ndim=1] energy = np.zeros(n, dtype=np.float64)
    #cdef double epsilon = 1e-10
    cdef double t = 2.0
    cdef double s = 2.0
    cdef double E0 = 3e-3
    cdef double nb = 2.0 / np.pi
    cdef double m_star = 0.01
    cdef double na = s * gamma(t / s) / gamma(1 / s) / gamma((t - 1) / s)
    cdef double fac = E0 * nb * na * f0 * np.sqrt(N0**2 - f0**2)
    cdef double KK, Ksqr, omm, c2star, ms, Atl
    for idx in range(n):
        #kkx = abs(kx[idx])
        #kky = abs(ky[idx])
        #kkz = abs(kz[idx])
        #if kkx+kky == 0.0:
        #    kkx = eps_h
        #    kky = eps_h
        #if kkz==0.0:
        #    kkz = eps_v
        kkx = abs(kx[idx]) + eps_h
        kky = abs(ky[idx]) + eps_h
        kkz = abs(kz[idx]) + eps_v
        KK = np.sqrt(kkx**2 + kky**2)
        Ksqr = KK**2 + kkz**2
        c2star = N0**2 / m_star**2
        om2 = (N0**2*KK**2 + f0**2*kkz**2 ) /(Ksqr)
        ms = (np.sqrt(N0**2 - om2) / c2star)
        Atl = ((1.0 + abs( kkz/ ms)**s)**(-t / s))
        if kx[idx]**2+ky[idx]**2+kz[idx]**2 == 0:
            energy[idx] = 0.0
        if kx[idx]**2+ky[idx]**2>(xmax)**2:
            energy[idx] = 0.0
        if kz[idx]**2>(zmax)**2:
            energy[idx] = 0.0
        else:
            energy[idx] = fac * Atl * (kkz)**2 / (4 * np.pi * ms) / ((N0**2 * KK**2 + f0**2 * kkz**2) * np.sqrt(Ksqr)) / (KK)
    return energy

# triad_sum function
def triad_sum(cnp.ndarray[double, ndim=1] kx0, cnp.ndarray[double, ndim=1] ky0, cnp.ndarray[double, ndim=1] kz0, 
              cnp.ndarray[double, ndim=1] om0, double s0, 
              cnp.ndarray[double, ndim=1] kx1, cnp.ndarray[double, ndim=1] ky1, cnp.ndarray[double, ndim=1] kz1, 
              cnp.ndarray[double, ndim=1] om1, double s1, 
              cnp.ndarray[double, ndim=1] kx2, cnp.ndarray[double, ndim=1] ky2, cnp.ndarray[double, ndim=1] kz2, 
              cnp.ndarray[double, ndim=1] om2, double s2, 
              double f0, double N0, double dt, double eps_h, double eps_v, double xmax, double zmax):
    cdef int n = kx0.shape[0]
    cdef cnp.ndarray[double, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[complex, ndim=1] C0, C1, C2
    cdef complex C3, Cx
    cdef cnp.ndarray[double, ndim=1] Ep_k0, Ep_k1, Ep_k2
    cdef double dom
    cdef complex icpmx = 1j
    C0 = coeff_C(kx0, ky0, kz0, om0, s0, kx1, ky1, kz1, om1, s1, kx2, ky2, kz2, om2, s2, f0, N0)
    C1 = coeff_C(kx2, ky2, kz2, om2, s2, -kx1, -ky1, -kz1, om1, -s1, kx0, ky0, kz0, om0, s0, f0, N0)
    C2 = coeff_C(kx1, ky1, kz1, om1, s1, -kx2, -ky2, -kz2, om2, -s2, kx0, ky0, kz0, om0, s0, f0, N0)
    Ep_k0 = Ep(kx0, ky0, kz0, om0, s0, f0, N0, eps_h, eps_v, xmax, zmax)
    Ep_k1 = Ep(kx1, ky1, kz1, om1, s1, f0, N0, eps_h, eps_v, xmax, zmax)
    Ep_k2 = Ep(kx2, ky2, kz2, om2, s2, f0, N0, eps_h, eps_v, xmax, zmax)
    for i in range(n):
        C3 = 2.0 * C0[i] * 4.0 * (Ep_k1[i] * Ep_k2[i] * C0[i].conjugate() - Ep_k0[i] * Ep_k1[i] * C1[i] - Ep_k0[i] * Ep_k2[i] * C2[i])
        dom = om1[i] + om2[i] - om0[i]
        Cx = icpmx * (1 - np.exp(icpmx * dom * dt)) / (dom if dom != 0 else 1e-24)
        result[i] = np.real(C3 * Cx)
    return result

def sum_integral(double X0, double Y0, double Z0,
                     cnp.ndarray[cnp.double_t, ndim=1] X, cnp.ndarray[cnp.double_t, ndim=1] Y, cnp.ndarray[cnp.double_t, ndim=1] Z,
                     double xmax, double ymax, double zmax, double f0, double N0, double dt, double dV, double eps_h, double eps_v):
    cdef:
        double wave_sum_
        double wave_sum_f_re, wave_sum_f_nr, wave_sum_N_re, wave_sum_N_nr
        double wave_sum_rest_re, wave_sum_rest_nr
        cnp.ndarray[cnp.double_t, ndim=1] X2, Y2, Z2
        object n
        cnp.ndarray[cnp.double_t, ndim=1] kx0, ky0, kz0, kx1, ky1, kz1, kx2, ky2, kz2
        cnp.ndarray[cnp.double_t, ndim=1] c_f, c_N, om0, om1, om2, wave_sum, reso, n_reso

    #if X0**2 + Y0**2 + Z0**2 == 0:
    #    return(X0, Y0, Z0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Compute X2, Y2, Z2
    X2 = X0 - X
    Y2 = Y0 - Y
    Z2 = Z0 - Z

    # Find indices satisfying the conditions
    n = np.nonzero((X2**2 + Y2**2 < xmax**2 + 1e-12) & (Z2**2 < zmax**2 + 1e-12) & (X2**2 + Y2**2 + Z2**2 > 0) & (X**2 + Y**2 + Z**2 > 0) & (X**2 + Y**2 < xmax**2 + 1e-12) )[0]

    if len(n) == 0:
        wave_sum_ = 0.0
        wave_sum_f_re = wave_sum_f_nr = wave_sum_N_re = wave_sum_N_nr = 0.0
        wave_sum_mix_re = wave_sum_mix_nr = 0.0
        wave_sum_rest_re = wave_sum_rest_nr = 0.0
    else:
        kx0 = np.full(len(X2), X0)[n]
        ky0 = np.full(len(Y2), Y0)[n]
        kz0 = np.full(len(Z2), Z0)[n]
        kx1 = X[n]
        ky1 = Y[n]
        kz1 = Z[n]
        kx2 = X2[n]
        ky2 = Y2[n]
        kz2 = Z2[n]

        n_f1 = (kx1**2 + ky1**2 == 0)
        n_f2 = (kx2**2 + ky2**2 == 0)
        n_N1 = (kz1 == 0)
        n_N2 = (kz2 == 0)
        n_f = (n_f1 & (n_N2 == 0)) | (n_f2 & (n_N1 == 0))
        n_N = (n_N1 & (n_f2 == 0)) | (n_N2 & (n_f1 == 0))
        n_mix = (n_f1 & n_N2) | (n_f2 & n_N1)

        #c_f = np.where((kx1**2 + ky1**2 == 0) | (kx2**2 + ky2**2 == 0), 1.0, 0.0)
        #c_N = np.where((kz1 == 0) | (kz2 == 0), 1.0, 0.0)
        om0 = np.sqrt(omega2_true(kx0, ky0, kz0, 1, f0, N0))
        om1 = np.sqrt(omega2_true(kx1, ky1, kz1, 1, f0, N0))
        om2 = np.sqrt(omega2_true(kx2, ky2, kz2, 1, f0, N0))
        wave_sum = triad_sum(kx0, ky0, kz0, om0, 1, kx1, ky1, kz1, om1, 1, kx2, ky2, kz2, om2, 1, f0, N0, dt, eps_h, eps_v, xmax, zmax)
        reso = np.where(np.abs(om1 + om2 - om0) < 0.0001, 1.0, 0.0)
        n_reso = 1.0 - reso
        wave_sum_ = np.nansum(np.where(n_f + n_N + n_mix== 0, wave_sum, 0.0))*dV
        wave_sum_f_re = np.nansum(reso * n_f * wave_sum) * dV
        wave_sum_f_nr = np.nansum(n_reso * n_f * wave_sum) * dV
        wave_sum_mix_re = np.nansum(reso * n_mix * wave_sum) * dV
        wave_sum_mix_nr = np.nansum(n_reso * n_mix * wave_sum) * dV
        wave_sum_N_re = np.nansum(reso * n_N * wave_sum) * dV
        wave_sum_N_nr = np.nansum(n_reso * n_N * wave_sum) * dV
        wave_sum_rest_re = np.nansum(reso * np.where(n_f + n_N + n_mix== 0, wave_sum, 0.0)) * dV
        wave_sum_rest_nr = np.nansum(n_reso * np.where(n_f + n_N + n_mix== 0, wave_sum, 0.0)) * dV
    
    return(X0, Y0, Z0, wave_sum_,wave_sum_f_re, wave_sum_f_nr, wave_sum_N_re, wave_sum_N_nr, wave_sum_mix_re, wave_sum_mix_nr, wave_sum_rest_re, wave_sum_rest_nr)



# triad_diff function
def triad_diff(cnp.ndarray[double, ndim=1] kx0, cnp.ndarray[double, ndim=1] ky0, cnp.ndarray[double, ndim=1] kz0, 
               cnp.ndarray[double, ndim=1] om0, double s0, 
               cnp.ndarray[double, ndim=1] kx1, cnp.ndarray[double, ndim=1] ky1, cnp.ndarray[double, ndim=1] kz1, 
               cnp.ndarray[double, ndim=1] om1, double s1, 
               cnp.ndarray[double, ndim=1] kx2, cnp.ndarray[double, ndim=1] ky2, cnp.ndarray[double, ndim=1] kz2, 
               cnp.ndarray[double, ndim=1] om2, double s2, 
               double f0, double N0, double dt, double eps_h, double eps_v, double xmax, double zmax):
    cdef int n = kx0.shape[0]
    cdef cnp.ndarray[double, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[complex, ndim=1] C0, C1, C2
    cdef complex C3, Cx
    cdef cnp.ndarray[double, ndim=1] Ep_k0, Ep_k1, Ep_k2
    cdef double dom
    cdef complex icpmx = 1j
    C0 = coeff_C(kx0, ky0, kz0, om0, s0, kx1, ky1, kz1, om1, s1, -kx2, -ky2, -kz2, om2, -s2, f0, N0)
    C1 = coeff_C(-kx2, -ky2, -kz2, om2, -s2, -kx1, -ky1, -kz1, om1, -s1, kx0, ky0, kz0, om0, s0, f0, N0)
    C2 = coeff_C(kx1, ky1, kz1, om1, s1, kx2, ky2, kz2, om2, s2, kx0, ky0, kz0, om0, s0, f0, N0)
    Ep_k0 = Ep(kx0, ky0, kz0, om0, s0, f0, N0, eps_h, eps_v, xmax, zmax)
    Ep_k1 = Ep(kx1, ky1, kz1, om1, s1, f0, N0, eps_h, eps_v, xmax, zmax)
    Ep_k2 = Ep(kx2, ky2, kz2, om2, s2, f0, N0, eps_h, eps_v, xmax, zmax)
    for i in range(n):
        C3 = 2.0 * C0[i] * 4.0 * (Ep_k1[i] * Ep_k2[i] * C0[i].conjugate() - Ep_k0[i] * Ep_k1[i] * C1[i] - Ep_k0[i] * Ep_k2[i] * C2[i])
        dom = om1[i] - om2[i] - om0[i]
        Cx = icpmx * (1 - np.exp(icpmx * dom * dt)) / (dom if dom != 0 else 1e-24)
        result[i] = np.real(C3 * Cx)
    return result

def diff_integral(double X0, double Y0, double Z0,
                     cnp.ndarray[cnp.double_t, ndim=1] X, cnp.ndarray[cnp.double_t, ndim=1] Y, cnp.ndarray[cnp.double_t, ndim=1] Z,
                     double xmax, double ymax, double zmax, double f0, double N0, double dt, double dV, double eps_h, double eps_v):
    cdef:
        double wave_diff_
        double wave_diff_f_re, wave_diff_f_nr, wave_diff_N_re, wave_diff_N_nr
        double wave_diff_rest_re, wave_diff_rest_nr
        cnp.ndarray[cnp.double_t, ndim=1] X2, Y2, Z2
        object n
        cnp.ndarray[cnp.double_t, ndim=1] kx0, ky0, kz0, kx1, ky1, kz1, kx2, ky2, kz2
        cnp.ndarray[cnp.double_t, ndim=1] c_f, c_N, om0, om1, om2, wave_sum, wave_diff, reso, n_reso

    #if X0**2 + Y0**2 + Z0**2 == 0:
    #    return(X0, Y0, Z0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Repeat for wave_diff
    X2 = -X0 + X
    Y2 = -Y0 + Y
    Z2 = -Z0 + Z
    n = np.nonzero((X2**2 + Y2**2 < xmax**2 + 1e-12) & (Z2**2 < zmax**2 + 1e-12) & (X2**2 + Y2**2 + Z2**2 > 0) & (X**2 + Y**2 + Z**2 > 0))[0]

    if len(n) == 0:
        wave_diff_ = 0.0
        wave_diff_f_re = wave_diff_f_nr = wave_diff_N_re = wave_diff_N_nr = 0.0
        wave_diff_rest_re = wave_diff_rest_nr = 0.0
    else:
        kx0 = np.full(len(X2), X0)[n]
        ky0 = np.full(len(Y2), Y0)[n]
        kz0 = np.full(len(Z2), Z0)[n]
        kx1 = X[n]
        ky1 = Y[n]
        kz1 = Z[n]
        kx2 = X2[n]
        ky2 = Y2[n]
        kz2 = Z2[n]

        n_f1 = (kx1**2 + ky1**2 == 0)
        n_f2 = (kx2**2 + ky2**2 == 0)
        n_N1 = (kz1 ==0)
        n_N2 = (kz2 ==0)
        n_f = (n_f1 & (n_N2 == 0)) | (n_f2 & (n_N1 == 0))
        n_N = (n_N1 & (n_f2 == 0)) | (n_N2 & (n_f1 == 0))
        n_mix = (n_f1 & n_N2) | (n_f2 & n_N1)

        #c_f = np.where((kx1**2 + ky1**2 == 0) | (kx2**2 + ky2**2 == 0), 1.0, 0.0)
        #c_N = np.where((kz1 == 0) | (kz2 == 0), 1.0, 0.0)
        om0 = np.sqrt(omega2_true(kx0, ky0, kz0, 1, f0, N0))
        om1 = np.sqrt(omega2_true(kx1, ky1, kz1, 1, f0, N0))
        om2 = np.sqrt(omega2_true(kx2, ky2, kz2, 1, f0, N0))
        wave_diff = triad_diff(kx0, ky0, kz0, om0, 1, kx1, ky1, kz1, om1, 1, kx2, ky2, kz2, om2, 1, f0, N0, dt, eps_h, eps_v, xmax, zmax)
        reso = np.where(np.abs(om1 - om2 - om0) < 0.0001, 1.0, 0.0)
        n_reso = 1.0 - reso
        wave_diff_ = np.nansum(np.where(n_f + n_N + n_mix== 0, wave_diff, 0.0))*dV
        wave_diff_f_re = np.nansum(reso * n_f * wave_diff) * dV *2.0
        wave_diff_f_nr = np.nansum(n_reso * n_f * wave_diff) * dV*2.0
        wave_diff_mix_re = np.nansum(reso * n_mix * wave_diff) * dV*2.0
        wave_diff_mix_nr = np.nansum(n_reso * n_mix * wave_diff) * dV*2.0
        wave_diff_N_re = np.nansum(reso * n_N * wave_diff) * dV*2.0
        wave_diff_N_nr = np.nansum(n_reso * n_N * wave_diff) * dV*2.0
        wave_diff_rest_re = np.nansum(reso * np.where(n_f + n_N + n_mix== 0, wave_diff, 0.0)) * dV*2.0
        wave_diff_rest_nr = np.nansum(n_reso * np.where(n_f + n_N + n_mix== 0, wave_diff, 0.0)) * dV*2.0
    return(X0, Y0, Z0, wave_diff_,wave_diff_f_re, wave_diff_f_nr, wave_diff_N_re, wave_diff_N_nr, wave_diff_mix_re, wave_diff_mix_nr, wave_diff_rest_re, wave_diff_rest_nr)
    #return (X0, Y0, Z0, wave_diff_f_re, wave_diff_f_nr, wave_diff_N_re, wave_diff_N_nr, wave_diff_rest_re, wave_diff_rest_nr)