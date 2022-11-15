import numpy as np
#from skhep.math.vectors import LorentzVector,  Vector3D

def jet_boost_to_Em(jet_par, jet_const, m_zero = 0.25, e_zero = 1.):
    
    jet = LorentzVector()
    jet.setptetaphim(jet_par[0], jet_par[1], jet_par[2], jet_par[3])
    
    L_const=[] # list of constituents as lorentz vector
    for c in jet_const:
        cst=LorentzVector()
        cst.setptetaphim(c[0], c[1], c[2], 0)
        L_const.append(cst)
    
    phi = jet.phi()
    bv = jet.boostvector
    bv.x = 0
    bv.y = 0

    jet_1d = jet.rotatez(-phi)
    jet_1d = jet_1d.boost(bv)
    for Lc in L_const:
        Lc.rotatez(-phi)
        Lc.boost(bv)

    ######################################################
    #         Rescale Mass, Boost to Ref Energy          #
    ######################################################

    m_rescale = m_zero/jet_1d.m
    jet_1d = m_rescale*jet_1d #Rescale Mass

    tvec_ref = LorentzVector()
    p_ref = jet_1d.vector
    p_ref = (np.sqrt(  (np.abs(np.square(e_zero) - np.square(jet_1d.m)))/(np.abs(np.square(jet_1d.e) - np.square(jet_1d.m))) ) )*p_ref
    tvec_ref.setptetaphim(p_ref[0], jet_1d.eta, jet_1d.phi(), jet_1d.m)

    bp_x = tvec_ref.boostvector.x


    beta = (jet_1d.e - (jet_1d.px/bp_x))/(jet_1d.p - (jet_1d.e/bp_x))

    jet_1d = jet_1d.boost(beta, 0, 0) #Boost jet to reference energy
    for Lc in L_const:
        Lc.boost(beta, 0, 0)

    for i, Lc in enumerate(L_const):
        jet_const[i][0]=Lc.pt
        jet_const[i][1]=Lc.eta
        jet_const[i][2]=Lc.phi()


def calorimeter_image(event, 
                      IMG_SIZE=40, 
                      phi_bonds=(-0.8, 0.8),
                      eta_bonds=(-0.8, 0.8),
                      obj_num=200):
    #calculate pT eta and phi
    end=obj_num*4
    px=event[1:end:4]
    py=event[2:end:4]
    pz=event[3:end:4]
    pT=(px**2+py**2)**0.5
    phi=np.arctan2(py, px)
    P=(px**2+py**2+pz**2)**0.5
    #theta=np.arctan2(pT, pz)
    eta=-0.5*np.log((P+pz)/(P-pz))
    eta[eta==np.inf]=0
    eta[eta==-np.inf]=0
    eta=np.nan_to_num(eta)
    
    # substract phi and eta of the hardest objectst from other:
    phi=phi-phi[0]
    eta=eta-eta[0]
    
    phi[phi<-np.pi]+=2*np.pi
    phi[phi>np.pi]-=2*np.pi
    
    #calculate the pT centroid
    phi_centroid=np.sum(pT*phi)/np.sum(pT)
    eta_centroid=np.sum(pT*eta)/np.sum(pT)

    phi-=phi_centroid
    eta-=eta_centroid
    
    phi[phi<-np.pi]+=2*np.pi
    phi[phi>np.pi]-=2*np.pi
    
    #calculate the pT centroid
    phi_centroid=np.sum(pT*phi)/np.sum(pT)
    eta_centroid=np.sum(pT*eta)/np.sum(pT)
    phi-=phi_centroid
    eta-=eta_centroid
    
    #calculate the moment of inertia tensor
    I_xx=np.sum(pT*(phi)**2)
    I_yy=np.sum(pT*eta**2)
    I_xy=np.sum(pT*eta*phi)
    I=np.array([[I_xx, I_xy], [I_xy, I_yy]])
    
    #calculate the major principal axis
    w, v=np.linalg.eig(I)
    if(w[0]>w[1]):
        major=0
    else:
        major=1

    #turn the immage 
    alpha=-np.arctan2(v[1, major], v[0, major])
    phi_new=phi*np.cos(alpha)-eta*np.sin(alpha)
    eta_new=phi*np.sin(alpha)+eta*np.cos(alpha)
    phi=phi_new
    eta=eta_new
    
    #flip the image according to the largest constituent
    q1=sum(pT[(phi>0)*(eta>0)])
    q2=sum(pT[(phi<=0)*(eta>0)])
    q3=sum(pT[(phi<=0)*(eta<=0)])
    q4=sum(pT[(phi>0)*(eta<=0)])
    indx=np.argmax([q1, q2, q3, q4])
    if indx==1:
        phi*=-1
    elif indx==2:
        phi*=-1
        eta*=-1
    elif indx==3:
        eta*=-1
       
    #create a calorimeter picture (pixelation)
    image=np.histogram2d(phi, eta, IMG_SIZE, [phi_bonds, eta_bonds], weights=pT)
    image=image[0]
    
    #small check of the image
    if np.sum(image)==0:
        raise NameError('Image is 0')
    if np.sum([image>0])>200:
        raise NameError('Too many non-zero pixels')
        
    return image/np.sum(image)

def distorted_calorimeter_image(event, 
                      IMG_SIZE=40, 
                      phi_bonds=(-0.8, 0.8),
                      eta_bonds=(-0.8, 0.8),
                      obj_num=200,
                      distortion=2):
    #calculate pT eta and phi
    end=obj_num*4
    px=event[1:end:4]
    py=event[2:end:4]
    pz=event[3:end:4]
    pT=(px**2+py**2)**0.5
    phi=np.arctan2(py, px)
    P=(px**2+py**2+pz**2)**0.5
    #theta=np.arctan2(pT, pz)
    eta=-0.5*np.log((P+pz)/(P-pz))
    eta[eta==np.inf]=0
    eta[eta==-np.inf]=0
    eta=np.nan_to_num(eta)
    
    # substract phi and eta of the hardest objectst from other:
    distortion_sigma_phi=(phi_bonds[1]-phi_bonds[0])/IMG_SIZE*distortion
    distortion_sigma_eta=(eta_bonds[1]-eta_bonds[0])/IMG_SIZE*distortion
    phi=phi-phi[0]+np.rnadom(scale=distortion_sigma_phi, size=phi.shape)
    eta=eta-eta[0]+np.rnadom(scale=distortion_sigma_eta, size=phi.shape)
    
    phi[phi<-np.pi]+=2*np.pi
    phi[phi>np.pi]-=2*np.pi
    
    #calculate the pT centroid
    phi_centroid=np.sum(pT*phi)/np.sum(pT)
    eta_centroid=np.sum(pT*eta)/np.sum(pT)

    phi-=phi_centroid
    eta-=eta_centroid
    
    phi[phi<-np.pi]+=2*np.pi
    phi[phi>np.pi]-=2*np.pi
    
    
    #calculate the pT centroid
    phi_centroid=np.sum(pT*phi)/np.sum(pT)
    eta_centroid=np.sum(pT*eta)/np.sum(pT)
    phi-=phi_centroid
    eta-=eta_centroid
    
    #calculate the moment of inertia tensor
    I_xx=np.sum(pT*(phi)**2)
    I_yy=np.sum(pT*eta**2)
    I_xy=np.sum(pT*eta*phi)
    I=np.array([[I_xx, I_xy], [I_xy, I_yy]])
    
    #calculate the major principal axis
    w, v=np.linalg.eig(I)
    if(w[0]>w[1]):
        major=0
    else:
        major=1

    #turn the immage 
    alpha=-np.arctan2(v[1, major], v[0, major])
    phi_new=phi*np.cos(alpha)-eta*np.sin(alpha)
    eta_new=phi*np.sin(alpha)+eta*np.cos(alpha)
    phi=phi_new
    eta=eta_new
    
    #flip the image according to the largest constituent
    q1=sum(pT[(phi>0)*(eta>0)])
    q2=sum(pT[(phi<=0)*(eta>0)])
    q3=sum(pT[(phi<=0)*(eta<=0)])
    q4=sum(pT[(phi>0)*(eta<=0)])
    indx=np.argmax([q1, q2, q3, q4])
    if indx==1:
        phi*=-1
    elif indx==2:
        phi*=-1
        eta*=-1
    elif indx==3:
        eta*=-1
       
    #create a calorimeter picture (pixelation)
    image=np.histogram2d(phi, eta, IMG_SIZE, [phi_bonds, eta_bonds], weights=pT)
    image=image[0]
    
    #small check of the image
    if np.sum(image)==0:
        raise NameError('Image is 0')
    if np.sum([image>0])>200:
        raise NameError('Too many non-zero pixels')
        
    return image/np.sum(image)
    
    

def calorimeter_image_ptethaphi(event, 
                      IMG_SIZE=40, 
                      phi_bonds=(-0.8, 0.8),
                      eta_bonds=(-0.8, 0.8),
                      obj_num=200):
    #calculate pT eta and phi
    end=obj_num*3
    pT=event[0:end:3]
    eta=event[1:end:3]
    phi=event[2:end:3]
    
    # substract phi and eta of the hardest objectst from others:
    phi=phi-phi[0]
    eta=eta-eta[0]
    
    phi[phi<-np.pi]+=2*np.pi
    phi[phi>np.pi]-=2*np.pi
    
    #calculate the pT centroid
    phi_centroid=np.sum(pT*phi)/np.sum(pT)
    eta_centroid=np.sum(pT*eta)/np.sum(pT)

    phi-=phi_centroid
    eta-=eta_centroid
    
    phi[phi<-np.pi]+=2*np.pi
    phi[phi>np.pi]-=2*np.pi
    
    #calculate the pT centroid
    phi_centroid=np.sum(pT*phi)/np.sum(pT)
    eta_centroid=np.sum(pT*eta)/np.sum(pT)
    phi-=phi_centroid
    eta-=eta_centroid
    
    while np.any(np.abs(phi)>np.pi):
        phi[phi<-np.pi]+=2*np.pi
        phi[phi>np.pi]-=2*np.pi
    
    #calculate the moment of inertia tensor
    I_xx=np.sum(pT*(phi)**2)
    I_yy=np.sum(pT*eta**2)
    I_xy=np.sum(pT*eta*phi)
    I=np.array([[I_xx, I_xy], [I_xy, I_yy]])
    
    #calculate the major principal axis
    w, v=np.linalg.eig(I)
    if(w[0]>w[1]):
        major=0
    else:
        major=1

    #turn the immage 
    alpha=-np.arctan2(v[1, major], v[0, major])
    phi_new=phi*np.cos(alpha)-eta*np.sin(alpha)
    eta_new=phi*np.sin(alpha)+eta*np.cos(alpha)
    phi=phi_new
    eta=eta_new
    
    #flip the image according to the largest constituent
    q1=sum(pT[(phi>0)*(eta>0)])
    q2=sum(pT[(phi<=0)*(eta>0)])
    q3=sum(pT[(phi<=0)*(eta<=0)])
    q4=sum(pT[(phi>0)*(eta<=0)])
    indx=np.argmax([q1, q2, q3, q4])
    if indx==1:
        phi*=-1
    elif indx==2:
        phi*=-1
        eta*=-1
    elif indx==3:
        eta*=-1
       
    #create a calorimeter picture (pixelation)
    image=np.histogram2d(phi, eta, IMG_SIZE, [phi_bonds, eta_bonds], weights=pT)
    image=image[0]
    
    #small check of the image
    if np.sum(image)==0:
        raise NameError('Image is 0')
    if np.sum([image>0])>200:
        raise NameError('Too many non-zero pixels')
        
    return image/np.sum(image)


def calorimeter_image_ptethaphi_no_center(event, 
                      IMG_SIZE=40, 
                      phi_bonds=(-0.8, 0.8),
                      eta_bonds=(-0.8, 0.8),
                      obj_num=200):
    #calculate pT eta and phi
    end=obj_num*3
    pT=event[0:end:3]
    eta=event[1:end:3]
    phi=event[2:end:3]
    
    phi[phi<-np.pi]+=2*np.pi
    phi[phi>np.pi]-=2*np.pi
    
    #calculate the moment of inertia tensor
    I_xx=np.sum(pT*(phi)**2)
    I_yy=np.sum(pT*eta**2)
    I_xy=np.sum(pT*eta*phi)
    I=np.array([[I_xx, I_xy], [I_xy, I_yy]])
    
    #calculate the major principal axis
    w, v=np.linalg.eig(I)
    if(w[0]>w[1]):
        major=0
    else:
        major=1

    #turn the immage 
    alpha=-np.arctan2(v[1, major], v[0, major])
    phi_new=phi*np.cos(alpha)-eta*np.sin(alpha)
    eta_new=phi*np.sin(alpha)+eta*np.cos(alpha)
    phi=phi_new
    eta=eta_new
    
    #flip the image according to the largest constituent
    q1=sum(pT[(phi>0)*(eta>0)])
    q2=sum(pT[(phi<=0)*(eta>0)])
    q3=sum(pT[(phi<=0)*(eta<=0)])
    q4=sum(pT[(phi>0)*(eta<=0)])
    indx=np.argmax([q1, q2, q3, q4])
    if indx==1:
        phi*=-1
    elif indx==2:
        phi*=-1
        eta*=-1
    elif indx==3:
        eta*=-1
       
    #create a calorimeter picture (pixelation)
    image=np.histogram2d(phi, eta, IMG_SIZE, [phi_bonds, eta_bonds], weights=pT)
    image=image[0]
    
    #small check of the image
    """
    if np.sum(image)==0:
        raise NameError('Image is 0')
    if np.sum([image>0])>200:
        raise NameError('Too many non-zero pixels')
    """
    
    return image/np.sum(image)