import numpy as np

def snell_angles(n_list, alpha):
    """
    Given a list of (complex) refractive indices n_list[0..M]
    and an incident angle alpha in radians in the first medium,
    return the array of transmitted angles theta_j in each medium
    using n0 sin(alpha) = n_j sin(theta_j).
    """
    n0 = n_list[0]
    # We assume alpha is real (the external incidence angle).
    # For complex n_j, numpy handles arcsin of complex values automatically,
    # but you must be mindful of branches. Usually this works well if
    # alpha is not too large. 
    sin_alpha = np.sin(alpha)
    theta = []
    for n in n_list:
        # Snell's law: n0 sin(alpha) = n_j sin(theta_j).
        # => theta_j = arcsin( (n0 / n_j) sin(alpha) )
        # This works even if n is complex, though the result may be complex.
        theta_j = np.arcsin((n0 / n) * sin_alpha)
        theta.append(theta_j)
    return np.array(theta)

def make_layer_matrix_s(n, d, k0, theta):
    """
    Characteristic matrix M_j for s-polarization in a layer with:
      - refractive index n (complex)
      - physical thickness d (um)
      - angle theta (possibly complex)
      - free-space wavenumber k0 = 2*pi / wavelength_in_vacuum
    The matrix is:
        M_j = [[cos(beta), i sin(beta)/eta_j],
               [i eta_j sin(beta), cos(beta)]]
    where
        beta_j = k0 * n * d * cos(theta)
        eta_j = n cos(theta)
    If d=0, this becomes the identity matrix.
    """
    beta = k0 * n * d * np.cos(theta)
    eta_j = n * np.cos(theta)
    # For zero thickness, sin(beta) ~ 0 => M_j ~ identity
    # (which automatically “removes” this layer from the stack).
    M11 = np.cos(beta)
    M12 = 1j * np.sin(beta) / eta_j
    M21 = 1j * eta_j * np.sin(beta)
    M22 = np.cos(beta)
    # Combine into a 2x2
    # We ensure these are complex arrays in case n or beta is complex.
    return np.array([[M11, M12],
                     [M21, M22]], dtype=complex)

def make_layer_matrix_p(n, d, k0, theta):
    """
    Characteristic matrix M_j for p-polarization in a layer with:
      - refractive index n (complex)
      - physical thickness d (um)
      - angle theta (possibly complex)
      - free-space wavenumber k0 = 2*pi / wavelength_in_vacuum
    The matrix is:
        M_j = [[cos(beta), i sin(beta)/eta_j],
               [i eta_j sin(beta), cos(beta)]]
    but here
        eta_j = (n cos(theta))^-1 for p-polarization
    or more precisely:
        eta_j = cos(theta) / (n * cos(theta)) in the usual non-magnetic case
               = cos(theta) / (n cos(theta)) = 1/n
    Actually, the more general formula for p is
        eta_j = n cos(theta_j) / (1 if mu_j=1 else mu_j?), etc.
    The main difference from s is that we use  eta_j^p = n cos(theta_j)  for s,
    but for p we effectively invert that. A common approach is:
        eta_j^p = n_j cos(theta_j) / (mu_j^1/2)
        eta_j^s = (n_j / mu_j^1/2) cos(theta_j).
    In non-magnetic media, mu_j=1, so for p:
        eta_j^p = n_j cos(theta_j) / 1 = n_j cos(theta_j)
        for s:
        eta_j^s = n_j cos(theta_j).
    Actually that suggests s and p have the same form if mu=1.
    However, the reflection formula at the boundaries differs.
    For the standard TMM approach, we typically define:
        Y_j = n_j cos(theta_j)   (for s)
        Y_j = cos(theta_j) / n_j (for p)
    so that the matrix is the same shape. 
    Let's do the latter. 
    """
    beta = k0 * n * d * np.cos(theta)
    # For p-polarization, the “admittance” is Y_j = cos(theta_j)/n_j
    Y_j = np.cos(theta) / n
    M11 = np.cos(beta)
    M12 = 1j * np.sin(beta) / Y_j
    M21 = 1j * Y_j * np.sin(beta)
    M22 = np.cos(beta)
    return np.array([[M11, M12],
                     [M21, M22]], dtype=complex)

def tmm_reflection(n_list, d_list_um, alpha_deg, wavelength_um, pol='s'):
    """
    Return the complex reflection amplitude r (from the top) for a stack
    described by:
       - n_list: length M+2, [n0, n1, ..., nM+1]
                 where n0 is incident medium, nM+1 is substrate
       - d_list_um: length M, the physical thickness (in microns) of each layer 1..M
       - alpha_deg: external incidence angle in degrees
       - wavelength_um: free-space wavelength in microns
       - pol: 's' or 'p'
    The code uses the standard Transfer-Matrix Method.
    
    1) Compute angles in each layer from Snell's law.
    2) Multiply the characteristic matrices of each layer.
    3) Apply boundary conditions at top and bottom to get reflection amplitude.
    """
    alpha = np.deg2rad(alpha_deg)
    # wave number in vacuum:
    k0 = 2 * np.pi / wavelength_um
    
    # angles in each medium:
    theta = snell_angles(n_list, alpha)  # array of length M+2
    
    # Build up the total characteristic matrix from top to bottom
    M = np.identity(2, dtype=complex)
    
    # For each real layer j = 1..M (i.e. skipping the incident medium = layer0, substrate=layer M+1)
    for j in range(1, len(n_list) - 1):
        n_j = n_list[j]
        d_j = d_list_um[j - 1]
        if pol.lower() == 's':
            M_j = make_layer_matrix_s(n_j, d_j, k0, theta[j])
        else:
            # p-polarization
            # for non-magnetic media, Y_j^p = cos(theta_j)/n_j
            M_j = make_layer_matrix_p(n_j, d_j, k0, theta[j])
        M = M @ M_j
    
    # Once we have M, we compute reflection from the top (layer 0) into M-layers-then-substrate
    # For s-polarization, the "admittance" is Y_j^s = n_j cos(theta_j).
    # For p-polarization, the "admittance" is Y_j^p = cos(theta_j)/n_j.
    
    # top (incident) medium
    n0 = n_list[0]
    if pol.lower() == 's':
        Y0 = n0 * np.cos(theta[0])
    else:
        Y0 = np.cos(theta[0]) / n0
    
    # substrate (last medium)
    n_sub = n_list[-1]
    if pol.lower() == 's':
        Y_sub = n_sub * np.cos(theta[-1])
    else:
        Y_sub = np.cos(theta[-1]) / n_sub
    
    # The reflection amplitude from the total matrix M is:
    #   r = (Y0*(M11 + M12*Y_sub) - (M21 + M22*Y_sub)) / (Y0*(M11 + M12*Y_sub) + (M21 + M22*Y_sub))
    M11, M12 = M[0, 0], M[0, 1]
    M21, M22 = M[1, 0], M[1, 1]
    
    num = Y0*(M11 + M12*Y_sub) - (M21 + M22*Y_sub)
    den = Y0*(M11 + M12*Y_sub) + (M21 + M22*Y_sub)
    r = num / den
    return r

def Contrast_TMM(laser_wavelength_nm, incident_angle_degrees, refractive_indices, thicknesses_nm):
    """
    Computes the reflectance contrast for a layered stack using the Transfer-Matrix Method.
    
    Parameters:
      laser_wavelength_nm: Laser wavelength in nm.
      incident_angle_degrees: Incident angle in degrees.
      refractive_indices: List of refractive indices [n0, n1, ..., nLast]
                         where n0 is incident medium, nLast is substrate.
      thicknesses_nm: List of film thicknesses in nm for the intermediate layers
                      (i.e. the same length as refractive_indices) - 2.
                      The first of these is the "TMD" film that we may vary.
    
    Returns:
      contrast_p, contrast_s, contrast_np:
         The reflectance contrasts for p-polarization, s-polarization,
         and their average (non-polarized).
    
    Contrast is defined relative to the "baseline" (the stack without the TMD film).
    i.e.:
        Contrast = - (R_baseline - R_full) / R_baseline
                 = (R_full - R_baseline) / R_baseline.
    """
    # Convert thickness from nm to microns:
    thicknesses_um = np.array(thicknesses_nm) * 1e-3
    wavelength_um = laser_wavelength_nm * 1e-3
    
    # Full stack reflection:
    r_p_full = tmm_reflection(refractive_indices, thicknesses_um,
                              incident_angle_degrees, wavelength_um, pol='p')
    R_p_full = np.abs(r_p_full)**2
    r_s_full = tmm_reflection(refractive_indices, thicknesses_um,
                              incident_angle_degrees, wavelength_um, pol='s')
    R_s_full = np.abs(r_s_full)**2
    
    # Baseline stack: remove the TMD film => skip index[1] & thickness[0].
    refractive_indices_base = [refractive_indices[0]] + refractive_indices[2:]
    thicknesses_base_nm = thicknesses_nm[1:]  # drop the first thickness
    thicknesses_base_um = np.array(thicknesses_base_nm) * 1e-3
    
    r_p_base = tmm_reflection(refractive_indices_base, thicknesses_base_um,
                              incident_angle_degrees, wavelength_um, pol='p')
    R_p_base = np.abs(r_p_base)**2
    r_s_base = tmm_reflection(refractive_indices_base, thicknesses_base_um,
                              incident_angle_degrees, wavelength_um, pol='s')
    R_s_base = np.abs(r_s_base)**2
    
    contrast_p = (R_p_full - R_p_base) / R_p_base
    contrast_s = (R_s_full - R_s_base) / R_s_base
    contrast_np = 0.5*(contrast_p + contrast_s)
    
    return contrast_p, contrast_s, contrast_np