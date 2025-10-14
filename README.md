# Scattering Integral for Nonlinear Energy Transfers (ScINET)
ScINET is a computational framework based on the kinetic equation to evaluate energy transfers from wave-wave interactions under weak nonlinearity in oceanic internal gravity wave fields as formulated in Sebastia Saez et al. (2025 a,b). This framework expands upon previous numerical methods to integrate the kinetic equation derived in a non-hydrostatic Boussinesq system, see Eden et al. (2019b). ScINET provides numerical methods to integrate the kinetic equation for finite times (ScINET-Genesis) and for infinite times (ScINET-Infinity).

## ScINET - Genesis
ScINET-Genesis integrates the kinetic equation for finite times, and therefore evalutes resonant (where the sum or difference of the frequency of two interacting waves equals the one of the generated wave) and non-resonant interactions without distinction.

* M1: The kinetic equation is integrated in an equidistant grid in wavenumber space, using positive and negative wavenumbers including zeros for any predefined energy density spectrum.
* M2: For isotropic energy density spectra, it is sufficient to evaluate the integral on the plane with zero meridional wavenumber in the zonal and vertical wavenumber space.

## ScINET - Infinity
* M3: ScINET-Infinity integrated the kinetic equation 

## Author
    * Pablo Sebastia Saez

## License
[MIT](LICENSE.txt)
