import numpy as np
def Fresnel_TMD(laser_wavelength_nm, incident_angle_degrees, TMD_layer_thickness_nm, SiO2_thickness_nm, TMD_index, SiO2_index, Si_index):
    
    # number of TMD layers
    N = np.arange(0,100,1)
    
    # thickness of TMD film
    d1 = TMD_layer_thickness_nm*N*1e-3 #convert to um
    
    # thickness of SiO2
    d2 = SiO2_thickness_nm*1e-3 #convert to um

    #laser wavelength
    lamb = laser_wavelength_nm*1e-3 #convert to um

    #incident angle
    alpha= incident_angle_degrees*(np.pi/180) # convert to radians

    # Sample Stack, refractive indices
    N0 = 1.0003         # air
    N1 = TMD_index      # TMDC
    N2 = SiO2_index     # SiO2 layer, can use 1.4689
    N3 = Si_index       # Si substrate, can use 3.872-0.01637j


    n0p=N0/np.cos(alpha);
    n1p=N1/(1-((np.sin(alpha))**2)/N1**2)**0.5
    n2p=N2/(1-((np.sin(alpha))**2)/N2**2)**0.5
    n3p=N3/(1-((np.sin(alpha))**2)/N3**2)**0.5

    #amplitude reflection coefficient
    p01p=(n0p-n1p)/(n0p+n1p); 
    p02p=(n0p-n2p)/(n0p+n2p);
    p12p=(n1p-n2p)/(n1p+n2p);
    p23p=(n2p-n3p)/(n2p+n3p);

    F1 = np.exp(-4*np.pi*N1*d1*((1-((np.sin(alpha))**2)/N1**2)**0.5)*1j/lamb) #SiO2 thickness
    F2 = np.exp(-4*np.pi*N2*d2*((1-((np.sin(alpha))**2)/N2**2)**0.5)*1j/lamb) #SiO2 thickness

    r0p=(p02p+p23p*F2)/(1+p02p*p23p*F2)
    r1p=(p01p+p01p*p12p*p23p*F2+p12p*F1+p23p*F1*F2)/(1+p12p*p23p*F2+p01p*p12p*F1+p01p*p23p*F1*F2)

    R0p=r0p*r0p.conj();
    R1p=r1p*r1p.conj();
    Cp=-1*(R0p-R1p)/R0p;

    n0s=N0*np.cos(alpha);
    n1s=N1*(1-((np.sin(alpha))**2)/N1**2)**0.5;
    n2s=N2*(1-((np.sin(alpha))**2)/N2**2)**0.5;
    n3s=N3*(1-((np.sin(alpha))**2)/N3**2)**0.5;

    p01s=(n0s-n1s)/(n0s+n1s); #amplitude reflection coefficient
    p02s=(n0s-n2s)/(n0s+n2s);
    p12s=(n1s-n2s)/(n1s+n2s);
    p23s=(n2s-n3s)/(n2s+n3s);

    F1=np.exp(-4*np.pi*N1*d1*((1-((np.sin(alpha))**2)/N1**2)**0.5)*1j/lamb) #SiO2 thickness
    F2=np.exp(-4*np.pi*N2*d2*((1-((np.sin(alpha))**2)/N2**2)**0.5)*1j/lamb) #SiO2 thickness

    r0s=(p02s+p23s*F2)/(1+p02s*p23s*F2)
    r1s=(p01s+p01s*p12s*p23s*F2+p12s*F1+p23s*F1*F2)/(1+p12s*p23s*F2+p01s*p12s*F1+p01s*p23s*F1*F2)

    R0s=r0s*r0s.conj()
    R1s=r1s*r1s.conj()
    Cs=-(R0s-R1s)/R0s

    R0np=(R0s+R0p)/2;
    R1np=(R1s+R1p)/2;
    Cnp=-(R0np-R1np)/R0np;
    
    return N, Cp, Cs, Cnp, R0p
