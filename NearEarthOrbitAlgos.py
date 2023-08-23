import numpy as np

# keplerian Algos 1-11

class Findc2c3: #algo-1 finding c2,c3 for universal kepler eqation
    """
    we define two variables chi and phi for global scope of conic orbit
    where chi = sqrt(semi_major)*arcsin((position/semi_major-1)/eccentricity )
    where phi = chi**2/semi_major
    """
    def __init__(self,phi):
        self.phi = phi
        self.k = 398600.4418 #earth GM
        sin,cos,sqrt,sinh,cosh = np.sin,np.cos,np.sqrt,np.sinh,np.cosh
        if phi>1e-6:
            c2,c3 = (1-cos(sqrt(phi)))/phi,(sqrt(phi)-sin(sqrt(phi)))/sqrt(phi**3)
        elif phi<-1e-6:
            c2,c3 = (1-cosh(sqrt(-phi)))/phi,(-sqrt(-phi)+sinh(sqrt(-phi)))/sqrt((-phi)**3)
        else :
            c2,c3 = 1/2,1/6
        self.c2,self.c3 = c2,c3
    def timeEquation(self,chi,r0,v0):
        r0,v0 = np.array([list(r0),list(v0)])
        dt = (1/np.sqrt(self.k))*(chi**3*self.c3+
            (r0.dot(v0)/np.sqrt(self.k))*self.c2*chi**2+
            np.linalg.norm(r0)*chi*(1-self.phi*self.c3))
        return dt
    def positionEquation(self,chi,r0,v0):
        """ r0 and v0 must be numpy array"""
        norm = np.linalg.norm
        r = (self.c2*chi**2+
             (r0.dot(v0)/np.sqrt(self.k))*(1-self.phi*self.c3)*chi+
             norm(r0)*(1-self.phi*self.c2))
        return r
class KepEqtnE: #algo-2 ecentric anomaly
    """
    using newton rapson method for series itration of Eccentric anomaly in relation to Mean anomaly
    """
    def __init__(self,meanAnomaly,eccentricity):
        self.M = meanAnomaly
        self.e = eccentricity
        if (np.pi< meanAnomaly) or (meanAnomaly<0) or (meanAnomaly >np.pi):
            eccentricAnomaly = meanAnomaly-eccentricity
        else:
            eccentricAnomaly = meanAnomaly+eccentricity
        self.E = eccentricAnomaly
    def EccentricAnomaly(self,tolerance=None):
        if not tolerance:
            tolerance = 1e-6
        E = self.E
        while True:
            En = E+(self.M-E+self.e*np.sin(E))/(1-self.e*np.cos(E))
            if abs(En-E)>tolerance:
                E = E+(self.M-E+self.e*np.sin(E))/(1-self.e*np.cos(E))
            else :
                break
        return E
class KepEqtnP: #algo-3 parabolic anomaly
    """ solving for  parabolic Anomaly with Baker's equation"""
    def __init__(self,dt,periabsis):
        self.k = 398600.4418
        self.dt = dt
        self.periabsis = periabsis
        self.Parabolic_meanMotion = 2*np.sqrt(self.k/periabsis**3)
        self.s = 0.5*np.arctan(2/(3*self.Parabolic_meanMotion*dt))
        self.w = np.arctan((np.tan(self.s))**(1/3))
        self.ParabolicAnomaly = 2/np.tan(2*self.w)
class KepEqtnH: #algo-4 hyperbolic anomaly
    """ finding hyperbolic anomaly with mean anomaly and eccentricity"""
    def __init__ (self,meanAnomaly,eccentricity):
        self.M = meanAnomaly
        self.e = eccentricity
    def sign(self,value):
            return +1 if value > 0 else -1
    def hyperbolicAnomaly(self,tolarence=None):
        if not tolarence:
            tolarence = 1e-6
        if self.e <1.6 :
            if (np.pi < self.M <0) or (self.M >np.pi):
                H = self.M-self.e
            else:
                H = self.M+self.e
        elif (self.e <3.6)  and (abs(self.M) >np.pi ):
            H =self.M - self.sign(self.M)*self.e
        else :
            H = self.M/(self.e-1)
        while True:
            Hn = H +(self.M-self.e*np.sinh(H)+H)/(self.e*np.cosh(H)-1)
            if abs(Hn-H) >tolarence:
                H = Hn
            else:
                break
        return H
class v2Anomalies: #algo-5 true anomalies to E,B,H
    """ from true anomaly we intend to find which class of orbital anomaly it belongs"""
    def __init__(self,eccentricity,trueAnomaly):
        self.e = eccentricity
        self.nu = trueAnomaly
    def v2ellipticalAnomaly(self):
        if self.e<1 :
            return np.arccos((self.e+np.cos(self.nu))/(1+self.e*np.cos(self.nu)))
        return None
    def v2parabolicAnomaly(self):
        if self.e == 1:
            return np.tan(self.nu/2)
        return None
    def v2hyperbolicAnomaly(self):
        if self.e >1:
            return np.arccosh((self.e+np.cos(self.nu))/(1+self.e*np.cos(self.nu)))
        return None
class Anomaly2v: # algo-6 E,B,H to true anomaly
    """ from any anomalies to true anomaly """
    def __init__(self,eccentricty):
        self.e = eccentricty
    def elliptical2TrueAnomaly(self,E):
        if self.e <1 :
            return np.arccos((np.cos(E)-self.e)/(1-self.e*np.cos(E)))
        return None
    def parabolic2TrueAnomaly(self,periabsis,position):
        if self.e ==1:
            return np.arccos((periabsis-position)/position)
        return None
    def hyperbolic2TrueAnomaly(self,H):
        if self.e >1:
            return np.cos((np.cosh(H)-self.e)/(1-self.e*np.cosh(H)))
        return None
class KEPLER: #algo-8 Finding new position and velocity in an orbit with time
    def __init__(self,r0,v0,dt):
        """pass r0,v0 as an numpy array"""
        self.k = 398600.4418 # for earth can change to different planets as needed
        self.r0 = r0
        self.v0 = v0
        self.dt = dt
        self.r0scala = np.linalg.norm(r0)
        self.v0scala = np.linalg.norm(v0)
        self.hamiltionian = 0.5*self.v0scala**2- self.k/self.r0scala
        self.semiMajor = -self.k/(2*self.hamiltionian)
        self.alpha = 1/self.semiMajor
        self.angularMomentum = np.cross(self.r0,self.v0)
        self.wscala = np.linalg.norm(self.angularMomentum)
        self.periabsis = self.wscala**2/self.k
    def sign(self,value):
        return +1 if value >0 else -1
    def EllipticalorCircleChi(self):
        if self.alpha> 1e-6:
            chi = np.sqrt(self.k)*self.dt*self.alpha
            return chi
        return None
    def ParabolaChi(self):
        if self.alpha < 1e-6:
            self.s = 0.5*np.arctan(1/3*self.dt*np.sqrt(self.k/self.periabsis**3))
            self.w = np.arctan((np.tan(self.s))**(1/3))
            chi = 2*np.sqrt(self.periabsis)/np.tan(2*self.w)
            return chi
        return None 
    def HyperbolaChi(self):
        if self.alpha < -1e-6:
            a  = 1/self.alpha
            chi = self.sign(self.dt)*np.sqrt(-a)*np.log(-2*self.k*self.alpha*self.dt/(self.r0.dot(self.v0)+self.sign(self.dt)*np.sqrt(-self.k*a)*(1-self.r0scala*self.alpha)) )
            return chi
        return None
    def getChi(self):
        if self.alpha>1e-6:
            return self.EllipticalorCircleChi()
        elif abs(self.alpha) <1e-6:
            return self.ParabolaChi()
        else:
            return self.HyperbolaChi()
    def stepChiPhiC2C3R(self,chi):
        k = 398600.4418
        alpha = self.alpha
        r0 = self.r0
        v0 = self.v0
        dt = self.dt
        phi = chi**2*alpha
        get = Findc2c3(phi)
        c2,c3 = get.c2, get.c3
        r = get.positionEquation(chi,r0,v0)
        dT = np.sqrt(k)*(dt-get.timeEquation(chi,r0,v0))
        return chi+ dT/r ,phi,c2,c3,r
    def getrv(self,tolarence = None):
        if not tolarence:
            tolarence = 1e-6
        chi = self.getChi()
        while True:
            ChiN,phi,c2,c3,r = self.stepChiPhiC2C3R(chi)
            if abs (ChiN-chi) > tolarence:
                chi = ChiN
            else :
                break
        f = 1-chi**2*c2/self.r0scala
        g = self.dt- chi**3*c3/np.sqrt(self.k)
        dfdt = np.sqrt(self.k)*chi*(phi*c3-1)/(r*self.r0scala)
        dgdt = 1-chi**2*c2/r
        rNew = f*self.r0 +g*self.v0
        vNew = dfdt*self.r0 + dgdt*self.v0
        return rNew,vNew
class RV2COE:#algo-9 from r,v in ECEI to orbital elements p,a,e,i,omega,w,anomaly
    def __init__(self,r,v): 
        (cross, dot, norm, deg,k) = (np.cross, np.dot, np.linalg.norm,(180/np.pi),398600.4418)
        self.r = r
        self.v = v
        self.rscla = norm(self.r)
        self.vscla = norm(self.v)
        self.vr = dot(self.r/self.rscla, self.v)
        self.vp = np.sqrt(self.vscla**2-self.vr**2)
        self.angular_momentum = cross(self.r, self.v)
        self.wscla = norm(self.angular_momentum)
        self.hamiltonian = 0.5*self.vscla**2-k/self.rscla
        self.node = cross([0, 0, 1], self.angular_momentum)
        self.nodescla = norm(self.node)
        self.eccentricity = (1/k)*((self.vscla**2-(k/self.rscla))*self.r-self.v*dot(self.r, self.v))
        self.eccscla = np.linalg.norm(self.eccentricity)#2
        self.semi_major = -k/(2*self.hamiltonian) if self.eccscla !=1 else np.inf #semi major 
        self.semi_minor = self.semi_major*np.sqrt(1-self.eccscla**2) #semi minor
        self.semi_latus_rectum = self.semi_major*(1-self.eccscla**2) if self.eccscla!=1 else self.wscla**2/k #1 
        self.inclination = np.arccos(self.angular_momentum[2]/self.wscla)#3 
        self.right_ascension = np.arccos(self.node[0]/self.nodescla)#4
        self.right_ascension = 6.283185307179586-self.right_ascension if self.node[1]<0 else self.right_ascension
        self.argument_of_periapsis = (np.arccos(np.dot(self.node, self.eccentricity)/(self.eccscla*self.nodescla)))#5
        self.argument_of_periapsis = 6.283185307179586-self.argument_of_periapsis if self.eccentricity[2]<0 else self.argument_of_periapsis
        self.true_anomaly = np.arccos(np.dot(self.eccentricity, self.r)/(self.eccscla*self.rscla))#6
        self.true_anomaly = 6.283185307179586-self.true_anomaly if np.dot(self.r,self.v)<0 else self.true_anomaly
    def getAnomaly(self):
        if (0<self.eccscla<1) and (self.inclination==0):#EllipticalEquatorial anomaly
            wTrue = np.arccos(self.eccentricity[0]/self.eccscla)
            return 6.283185307179586-wTrue if self.eccentricity[1]<0 else wTrue
        elif (self.eccscla==0) and (0<self.inclination<np.pi): #CircularInclined Anomaly
            u = np.arccos(self.node.dot(self.r)/(self.nodescla*self.rscla))
            return 6.283185307179586-u if self.r[2]<0 else u
        elif (self.eccscla ==0 and self.inclination==0): #circular equatorial Anomaly
            lambdaTrue =np.arccos(self.r[0]/self.rscla)
            return 6.283185307179586-lambdaTrue if self.r[1]<0 else lambdaTrue
        return self.true_anomaly 
    def getOrbitalElements(self):
        return (
            self.semi_latus_rectum,
            self.eccscla,
            self.inclination,
            self.right_ascension,
            self.argument_of_periapsis,
            self.getAnomaly())
class COE2rv:#algo-10 from Orbital Elements to r,v
    def __init__(self,eccencricity,
            inclination,
            rightAscension,
            argumentOfPeriapsis,
            trueAnomaly,
            semiMajor=None,
            semiLatusRectum=None):
        self.k = 398600.4418
        self.eccscla = eccencricity
        self.inclination = inclination
        self.semiMajor = semiMajor
        self.right_ascension = rightAscension
        self.periapsis = argumentOfPeriapsis if self.semiMajor is None else self.semiMajor*(1-self.eccscla**2)
        self.semiLatusRectum = semiLatusRectum
        self.trueAnomaly = trueAnomaly
        self.r_PQW = np.array([self.semiLatusRectum*np.cos(self.trueAnomaly)/(1+self.eccscla*np.cos(self.trueAnomaly)),
                               self.semiLatusRectum*np.sin(self.trueAnomaly)/(1+self.eccscla*np.cos(self.trueAnomaly)),
                               0]) #1
        c = np.sqrt(self.k/semiLatusRectum)
        self.v_PQW = np.array([-c*np.sin(self.trueAnomaly),c*(self.eccscla+np.cos(self.trueAnomaly)),0])#2
        
        self.perifocal2ECEIMatrix = self.trasnformPerifocal2ECEI()
        self.rECEI = self.perifocal2ECEIMatrix.dot(self.r_PQW.T) #3
        self.vECEI = self.perifocal2ECEIMatrix.dot(self.v_PQW.T) #4
    def trasnformPerifocal2ECEI(self):
        sin,cos = np.sin,np.cos
        return np.array([
                    [cos(self.right_ascension)*cos(self.periapsis)-sin(self.right_ascension)*sin(self.periapsis)*cos(self.inclination),
                    -cos(self.right_ascension)*sin(self.periapsis)-sin(self.right_ascension)*cos(self.periapsis)*cos(self.inclination),
                    sin(self.right_ascension)*sin(self.inclination)],

                    [sin(self.right_ascension)*cos(self.periapsis)+cos(self.right_ascension)*sin(self.periapsis)*cos(self.inclination),
                    -sin(self.right_ascension)*sin(self.periapsis)+cos(self.right_ascension)*cos(self.periapsis)*cos(self.inclination),
                    -cos(self.right_ascension)*sin(self.inclination)],

                    [sin(self.periapsis)*sin(self.inclination),cos(self.periapsis)*sin(self.inclination),cos(self.inclination)]]) 
    def getrvPerifocal(self):
        return self.r_PQW,self.v_PQW
    def getrvECEI(self):
        return self.rECEI,self.vECEI
class FindTOF:#algo-11 time of flight 
    def __init__(self,r0,r,semiParameter):
        """r0,r must be numpy array"""
        self.k = 398600.4418
        self.r0,self.r ,self.semiParameter= r0,r,semiParameter
        self.r0s ,self.rs = np.linalg.norm(self.r0),np.linalg.norm(self.r)
        self.changeTrueAnomaly = np.arccos((self.r0.dot(self.r))/(self.r0s*self.rs))
        self.K = self.r0s*self.rs*(1-np.cos(self.changeTrueAnomaly))
        self.l = self.r0s+self.rs
        self.m = self.r0s*self.rs*(1+np.cos(self.changeTrueAnomaly))
        self.a = (self.m*self.K*self.semiParameter)/((2*self.m-self.l**2)*self.semiParameter**2+2*self.K*self.l*self.semiParameter-self.K**2)
        self.f = 1 - (self.rs/self.semiParameter)*(1-np.cos(self.changeTrueAnomaly))
        self.g = self.r0s*self.rs*np.sin(self.changeTrueAnomaly)/(np.sqrt(self.k*self.semiParameter))
    def EllipticalTOF(self):
        if self.a >0:
            changeEccentricAnomaly = np.arccos(1-(self.r0s/self.a)*(1-self.f))
            tof = self.g+np.sqrt(self.a**3/self.k)*(changeEccentricAnomaly-np.sin(changeEccentricAnomaly))
            return tof
        return None
    def ParabolicTOF(self):
        if self.a == np.inf:
            c = np.sqrt(self.r0s**2+self.rs**2-self.r0s*self.rs*np.cos(self.changeTrueAnomaly))
            s = (self.r0s+self.rs+c)/2
            tof = (2/3)*np.sqrt(s**3/(2*self.k))*(1-((s-c)/s)**(3/2))
            return tof
        return None
    def HyperbolicTOF(self):
        if self.a <0:
            changeHyperbolicAnomaly = np.arccosh(1+(self.f-1)*(self.r0s/self.a))
            tof = self.g+np.sqrt((-self.a)**3/self.k)*(np.sinh(changeHyperbolicAnomaly)-changeHyperbolicAnomaly)
        return None
    def getTOF(self):
        if self.a >0:
            return self.EllipticalTOF()
        elif self.a == np.infty:
            return self.ParabolicTOF()
        elif self.a <0:
            return self.HyperbolicTOF()


# initial orbit determination algos 51-61
class siteTrackinECEF: #algo-51 find position and velocity from ground observations in ECEF coordinates 
    def __init__(self,phi,Lambda,h_ellp,rho,beta,el,drhodt,dbetadt,deldt,year,month,day,utc,dUTl,dAT,xp,yp) :
        """
        phi = geodetic latitude
        Lambda = longitude
        h__ellp = elivation
        rho = distance of SAT from surface of Earth
        beta = angle
        el = evlivation angle
        """
        (self.phi,self.Lambda,self.h_ellp,self.rho,self.beta,self.el,self.drhodt,self.dbetadt,
        self.deldt,self.year,self.month,self.day,self.utc,self.dutl,self.dat,self.xp,self.yp ) =(
        phi,Lambda, h_ellp,rho,beta,el,drhodt,dbetadt,deldt,year,month,day,utc,dUTl,dAT,xp,yp)
        self.radiusEarth = 6378.137 #km
        self.eccenEarth = 0.081819221456
        self.CEarth = self.radiusEarth/(np.sqrt(1-self.eccenEarth**2*np.sin(phi)**2))
        self.sEarth = self.CEarth*(1-self.eccenEarth**2)
        self.rd = (self.CEarth+h_ellp)*np.cos(phi)
        self.rk = (self.sEarth+h_ellp)*np.sin(phi)
    def rsiteECEF(self):
        return np.array([
            [self.rd*np.cos(self.Lambda)],
            [-self.rd*np.sin(self.Lambda)],
            [self.rk]])
    def rhoSEZ(self):
        rho,el,beta = self.rho,self.el,self.beta
        return np.array([
            [-rho*np.cos(el)*np.cos(beta)],
            [rho*np.cos(el)*np.sin(beta)],
            [rho*np.sin(el)]])
    def drhodtSEZ(self):
        drhodt,el,beta,rho = self.drhodt,self.el,self.beta,self.rho
        deldt = self.deldt
        dbetadt = self.dbetadt
        return np.array([
            [-drhodt*np.cos(el)*np.cos(beta)+rho*np.sin(el)*np.cos(beta)*deldt+rho*np.cos(el)*np.sin(beta)*dbetadt],
            [drhodt*np.cos(el)*np.sin(beta)-rho*np.sin(el)*np.sin(beta)*deldt+rho*np.cos(el)*np.cos(beta)*dbetadt],
            [drhodt*np.sin(el)+rho*np.cos(el)*deldt]])
    def SEZ2ECEF(self):#check if the transformation is correct
        sin,cos,Lambda,phi= np.sin,np.cos,self.Lambda,self.phi
        return np.array([
        [sin(phi)*cos(Lambda),sin(Lambda),cos(phi)*cos(Lambda)],
        [-sin(phi)*sin(Lambda),cos(Lambda),-cos(phi)*sin(Lambda)],
        [-cos(phi),0,sin(phi)]])
    def rhoECEF(self):
        return self.SEZ2ECEF().dot(self.rhoSEZ())
    def vECEF(self):
        return self.SEZ2ECEF().dot(self.drhodtSEZ())
    def rECEF(self):
        return self.rhoECEF()+self.rsiteECEF()
class AnglesOnlyGauss:#algo-52 INCOMPLETE AND WRONG RESOLVE THIS 
    def __init__(self,L1,L2,L3,JD1,JD2,JD3,rsite1,rsite2,rsite3):
        self.L1,self.L2,self.L3 = L1,L2,L3
        self.jd1,self.jd2,self.jd3 = JD1,JD2,JD3
        self.r1,self.r2,self.r3 = rsite1,rsite2,rsite3
        self.tau1,self.tau3 = JD1-JD2,JD3-JD2
        self.a1 = self.tau3/(self.tau3-self.tau1)
        self.a1u = (self.tau3*((self.tau3-self.tau1)**2-self.tau3**2))/(6*(self.tau3-self.tau1))
        self.a3 = -self.tau1/(self.tau3-self.tau1)
        self.a3u = (self.tau1*((self.tau3-self.tau1)**2-self.tau1**2))/(6*(self.tau3-self.tau1))
        Lx1,Ly1,Lz1 = L1
        Lx2,Ly2,Lz2 = L2
        Lx3,Ly3,Lz3 = L3
        self.L = np.array([
                [Ly2*Lz3-Ly3*Lz2, -Ly1*Lz3+Ly3*Lz1, Ly1*Lz2-Ly2*Lz1],
                [Lx2*Lz3-Lx3*Lz2, -Lx1*Lz3+Lx3*Lz1, Lx1*Lz2-Lx2*Lz1],
                [Lx2*Ly3-Lx3*Ly2, -Lx1*Ly3+Lx3*Ly1, Lx1*Ly2-Lx2*Ly1]])
        self.Linv = np.linalg.inv(self.L)
        self.M = [self.Linv.dot(rsite1),self.Linv.dot(rsite2),self.Linv.dot(rsite3)]
class GIBBS:#algo-54 find v2 from r1,r2,r3 where they in ECI
    def __init__(self,r1,r2,r3):
        """ r1,r2,r3 must be in numpy array"""
        self.k = 398600.4418
        norm = np.linalg.norm
        rs1,rs2,rs3 = norm(r1),norm(r2),norm(r3)
        z12 = np.cross(r1,r2)
        z23 = np.cross(r2,r3)
        z31 = np.cross(r3,r1)
        self.N = rs1*z23+rs2*z31+rs3*z12
        self.D = z12+z23+z31
        self.S = (rs2-rs3)*r1+(rs3-rs1)*r2+(rs1-rs2)*r3
        self.B = np.cross(self.D,r2)
        self.Lg = np.sqrt(self.k/(norm(self.N)*norm(self.D)))
        self.v2 = (self.Lg/rs2)*self.B+self.Lg*self.S
        
class herrickGIBBS:#algo-55 same as GIBBS but we use juliean Date to find v2
    def __init__(self,r1,r2,r3,JD1,JD2,JD3) :
        """positions r must be numpy array"""
        norm = np.linalg.norm
        rs1,rs2,rs3 =norm(r1),norm(r2),norm(r3)
        k = 398600.4418
        self.T31 = JD3-JD1
        self.T32 = JD3-JD2
        self.T21 = JD2-JD1
        self.v2 = (-self.T32*((1/(self.T21*self.T31))+(k/(12*rs1**3)))*r1
                +(self.T32-self.T21)*((1/(self.T21*self.T32))+(k/(12*rs2**3)))*r2
                +self.T21*((1/(self.T32*self.T31))+(k/(12*rs3**3)))*r3)
class LambertsMinEnergy:#algo-56 find a,e,t,v0 minimum from r0,r
    def __init__(self,r0,r):
        """r0 ->initial position must be numpy array
           r -> any step position must be in numpy array 
        """
        sin,cos,acos,norm = np.sin,np.cos,np.arccos,np.linalg.norm
        self.r0,self.r = r0,r
        self.rs0,self.rs = norm(r0),norm(r)
        self.dtrueAnomaly = acos((np.dot(r0,r))/(self.rs0*self.rs))
    def minimumEnergy(self):
        """a,e,tabsolute,tamin,v0"""
        k = 398600.4418
        sin,cos,acos,norm = np.sin,np.cos,np.arccos,np.linalg.norm
        c = np.sqrt(self.rs0**2+self.rs**2-2*self.rs*self.rs0*cos(self.dtrueAnomaly))
        s = 0.5*(self.rs0+self.rs+c)
        beta = 2*np.arcsin(np.sqrt((s-c)/s))
        semiMajorMin = s/2
        semiLatusRectumMin = (self.rs*self.rs0/c)*(1-cos(self.dtrueAnomaly))
        eccentricityMin = np.sqrt(1-(2*semiLatusRectumMin/s))
        taMin = np.sqrt(semiMajorMin**3/k)*(np.pi+(beta-sin(beta)))
        tabsoluteMin = 1/3*(np.sqrt(2/k)*(s**(3/2)-(s-c)**(3/2)))
        v0 = np.sqrt(k*semiLatusRectumMin)/(self.rs*self.rs0*sin(self.dtrueAnomaly))*(self.r-(1-((self.rs/semiLatusRectumMin)*(1-cos(self.dtrueAnomaly))))*self.r0)
        return (semiMajorMin,eccentricityMin,tabsoluteMin,taMin,v0)        
class HitEarth:#algo-60 min distance that would hit earth True or False
    def __init__(self,rInt,rTgt,vA,vB) :
        """all vectors must be in numpy array """
        self.rint,self.rtgt,self.va,self.vb,self.a = rInt,rTgt,vA,vB
    def collision(self):
        norm = np.linalg.norm
        k = 398600.4418
        if np.dot(self.rint,self.va) <0 and np.dot(self.rtgt,self.vb)>0:
            Energy = 0.5*self.va**2-k/norm(self.rint)
            semiMajor = -k/(2*Energy)
            AngularMomentum = norm(np.cross(self.rint,self.va))
            semilatusrectum = AngularMomentum**2/k
            eccentricity = np.sqrt((semiMajor-semilatusrectum)/semiMajor)
            rhit = semiMajor(1-eccentricity)
            if rhit<=6378.136: #radius of earth
                return True
            else:
                return False    

# special Perturbation Techniques algos 62-65
class PKepler:#algo-65 with initial position and velocity,mean motion rate and mean motion acceleration we determine Pertubrated KEPLER
    def __init__(self,r0,v0,dt,dn0dt,d2n0dt2):
        OrbitalElements = RV2COE(r0,v0)
        (self.semi_latus_rectum,
        self.eccscla,
        self.inclination,
        self.right_ascension,
        self.argument_of_periapsis,
        self.Anomaly)=OrbitalElements.getOrbitalElements()
        self.J2 = 1.082626925638815e-3 # The second zonal harmonic coefficient of the Earth
        self.rEarth = 6371 #radius of earth in km
        self.semiMajor = OrbitalElements.semi_major
        self.EllipAnomaly = OrbitalElements.EllipAnomaly if self.eccscla!=0 else self.Anomaly
        self.MeanMotion = OrbitalElements.MeanMotion
        self.MeanAnomaly = OrbitalElements.MeanAnomaly # n 
        self.PsemiMajor = self.semiMajor-(2*self.semiMajor*dn0dt*dt)/(3*self.MeanMotion)
        self.Peccscla = self.eccscla-(2*(1-self.eccscla)*dn0dt*dt)/(3*dn0dt)
        self.PsemiLatusRectum = self.PsemiMajor*(1-self.Peccscla**2)
        self.PmeanAnomaly = self.MeanAnomaly+self.MeanMotion*dt**2+0.5*dn0dt*dt+d2n0dt2*dt**3/6
        self.PrightAscension = self.right_ascension-(3*self.MeanMotion*self.rEarth*2*self.J2*np.cos(self.inclination)*dt)/(2*self.semi_latus_rectum**2)
        self.PargumenOfPeriapsis = self.argument_of_periapsis+dt*(4-5*np.sin(self.inclination)**2)*(3*self.MeanMotion*self.rEarth**2*self.J2)/(4*self.semi_latus_rectum**2)
        self.PeccentricAnomaly = KepEqtnE(self.MeanAnomaly,self.Peccscla).EccentricAnomaly()
        self.Panomaly = Anomaly2v(self.Peccscla).elliptical2TrueAnomaly(self.PeccentricAnomaly) if self.eccscla!=0 else self.PeccentricAnomaly
        self.PrECEI,self.PvECEI = self.getPertubratedRV()
    def updatedPertubrationsOfOrbitalElements(self):# update for pertubrations
        return (self.PsemiLatusRectum,
                self.Peccscla,
                self.inclination,
                self.PrightAscension,
                self.PargumenOfPeriapsis,
                self.Panomaly)
    def getPertubratedRV(self):
        return COE2rv(eccencricity=self.Peccscla,
                      inclination=self.inclination,
                      rightAscension=self.PrightAscension,
                      argumentOfPeriapsis=self.PargumenOfPeriapsis,
                      trueAnomaly=self.Panomaly,
                      semiLatusRectum=self.PsemiLatusRectum).getrvECEI()  
# Orbit determinatin and estimation algos 66-70

# Mission Analysis algos 71-76
