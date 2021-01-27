
#################################################################################################
#################################################################################################
#################################################################################################

## rotationmatrix calculates the matrix that defines theta rotation about the r axis
def rotationmatrix(r,theta):
    #convert to unit vector
    r = np.squeeze(np.asarray(r))
    r = r / np.sqrt(r @ r.T)
    
    ux, uy, uz = r;  #obtain each element
    
    #each rows of the matrix
    row1 = [np.cos(theta)+(ux**2*(1-np.cos(theta))),
            (ux*uy*(1-np.cos(theta)))-(uz*np.sin(theta)),
            (ux*uz*(1-np.cos(theta)))+(uy*np.sin(theta))
           ]
    row2 = [(uy*ux*(1-np.cos(theta)))+(uz*np.sin(theta)),
            np.cos(theta)+(uy**2*(1-np.cos(theta))),
            (uy*uz*(1-np.cos(theta)))-(ux*np.sin(theta))
           ]
    row3 = [(uz*ux*(1-np.cos(theta)))-(uy*np.sin(theta)),
            (uz*uy*(1-np.cos(theta)))+(ux*np.sin(theta)),
            np.cos(theta) + (uz**2*(1-np.cos(theta)))
           ]
    
    #the rotation amtrix
    R = np.array([row1,row2,row3]);
    del row1, row2, row3
    
    # custom threshold for 0
    thresh_i = np.absolute(R) < 10**-9;
    R[thresh_i] = 0.;
    
    #return the rotation matrix
    return R

#################################################################################################
#################################################################################################
#################################################################################################

## anglebetweenlines calculates angle between two lines with slope m1 and m2
def angleTBs(rPm, k1_m,k2_m):
    # plane normal in reference frame
    k1_r = k1_m @ inv(rPm)
    k2_r = k2_m @ inv(rPm)
    
    # slope of the twin boundary line in [x,y]ᵣ
    m1 = -k1_r[0]/k1_r[1]
    m1 = -k1_r[0]/k1_r[1]
    
    #phi is the angle between two lines with slope m1 and m2
    phi = np.arctan((m1-m2)/(1+m1*m2))
    
    #return phi in degrees
    return np.degrees(phi)

#################################################################################################
#################################################################################################
#################################################################################################