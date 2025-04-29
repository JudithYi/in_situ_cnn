import numpy as np
import sys
import numba 
def progressbar(current_value,total_value,bar_lengh,progress_char,progress_name="Progress"): 
    percentage = int((current_value/total_value)*100)                                                # Percent Completed Calculation 
    progress = int((bar_lengh * current_value ) / total_value)                                       # Progress Done Calculation 
    loadbar = progress_name+": [{:{len}}]{}%".format(progress*progress_char,percentage,len = bar_lengh)    # Progress Bar String
    print(loadbar, end='\r')     
spec = [
    ('pixels', numba.float64[:]),
    ('GxArray', numba.float64[:]),
    ('GyArray', numba.float64[:]),
    ('radArray', numba.float64[:]),
    ('SRRFArray', numba.float64[:]),
    ('xRingCoordinates0', numba.float64[:]),
    ('yRingCoordinates0', numba.float64[:]),
    ('xRingCoordinates1', numba.float64[:]),
    ('yRingCoordinates1', numba.float64[:]),
    ('shiftX', numba.float64[:]),
    ('shiftY', numba.float64[:]),
    ('magnification', numba.int32),
    ('SRRForder', numba.int32),
    ('nRingCoordinates', numba.int32),
    ('border', numba.int32),
    ('ParallelGRadius', numba.int32),
    ('PerpendicularGRadius', numba.int32),
    ('symmetryAxis', numba.int32),
    ('display', numba.int32),
    ('width', numba.int32),
    ('height', numba.int32),
    ('widthHeight', numba.int32),
    ('nTimePoints', numba.int32),
    ('widthHeightTime', numba.int32),
    ('widthM', numba.int32),
    ('heightM', numba.int32),
    ('widthHeightM', numba.int32),
    ('widthHeightMTime', numba.int32),
    ('borderM', numba.int32),
    ('widthMBorderless', numba.int32),
    ('heightMBorderless', numba.int32),
    ('widthHeightMBorderless', numba.int32),
    ('widthHeightMTimeBorderless', numba.int32),
    ('spatialRadius', numba.float64),
    ('gradRadius', numba.float64),
    ('doRadialitySquaring', numba.boolean),
    ('renormalize', numba.boolean),
    ('doIntegrateLagTimes', numba.boolean),
    ('radialityPositivityConstraint', numba.boolean),
    ('doGradWeight', numba.boolean),
    ('doIntensityWeighting', numba.boolean),
    ('doGradSmooth', numba.boolean),
    ('setupComplete', numba.boolean),
]

@numba.experimental.jitclass(spec)
class SRRF:

    def __init__(self):
        self.magnification = 0
        self.SRRForder = 0
        self.nRingCoordinates = 0
        self.spatialRadius = 0.0
        self.gradRadius = 0.0
        self.border = 0
        self.doRadialitySquaring = True
        self.renormalize = False
        self.doIntegrateLagTimes = True
        self.radialityPositivityConstraint = False
        self.doGradWeight = False
        self.doIntensityWeighting = False
        self.doGradSmooth = False
        self.ParallelGRadius = 0
        self.PerpendicularGRadius = 0
        self.display = 0
        self.setupComplete = False
    
    def setupSRRF(self, \
                  magnification, SRRForder, symmetryAxis,   \
                  spatialRadius, psfWidth, border,      \
                  doRadialitySquaring, renormalize, doIntegrateLagTimes, \
                  radialityPositivityConstraint,        \
                  doGradWeight, doIntensityWeighting, doGradSmoothing,   \
                  display_name ):
        if self.setupComplete == True:
            print("ERROR: SRRF is setup once")
            return
        self.magnification = magnification
        self.SRRForder = SRRForder
        self.nRingCoordinates = symmetryAxis * 2
        self.spatialRadius = spatialRadius * np.float64(self.magnification)
        self.gradRadius = psfWidth * np.float64(self.magnification)
        self.border = border
        self.doRadialitySquaring = doRadialitySquaring
        self.renormalize = renormalize
        self.doIntegrateLagTimes = doIntegrateLagTimes
        self.radialityPositivityConstraint = radialityPositivityConstraint
        self.doGradWeight = doGradWeight
        self.doIntensityWeighting = doIntensityWeighting
        self.doGradSmooth = doGradSmoothing
        if(self.doGradSmooth):
            self.ParallelGRadius = 2
            self.PerpendicularGRadius = 1
        else:
            self.ParallelGRadius = 1
        if display_name == "Gradient":
            self.display = 1
        elif display_name == "Gradient Ring Sum":
            self.display = 2
        elif display_name == "Intensity Interp":
            self.display = 3
        self.setupComplete = True
    
    def calculate(self, pictures, width, height, \
                  shiftX, shiftY):
        # Input image properties
        self.width = width
        self.height = height
        self.widthHeight = width * height
        self.nTimePoints = len(pictures)
        self.widthHeightTime = self.widthHeight * self.nTimePoints

        # Radiality Image Properties
        self.widthM = width * self.magnification
        self.heightM = height * self.magnification
        self.widthHeightM = self.widthM * self.heightM
        self.widthHeightMTime = self.widthHeightM * self.nTimePoints
        self.borderM = self.border * self.magnification
        self.widthMBorderless = self.widthM - self.borderM * 2
        self.heightMBorderless = self.heightM - self.borderM * 2
        self.widthHeightMBorderless = self.widthMBorderless * self.heightMBorderless
        self.widthHeightMTimeBorderless = self.widthHeightMBorderless * self.nTimePoints

        # Initialise Arrays
        self.pixels = np.reshape(pictures, (self.widthHeightTime))
        self.GxArray = np.zeros(self.widthHeightTime, dtype=np.float64)
        self.GyArray = np.zeros(self.widthHeightTime, dtype=np.float64)
        self.radArray = np.zeros(self.widthHeightMTime, dtype=np.float64)
        self.SRRFArray = np.zeros(self.widthHeightM, dtype=np.float64)
        self.shiftX = shiftX
        self.shiftY = shiftY

    def calculateSRRF(self,idx):
        x = int(int(idx % int(self.width*1.75)) / 1.75)
        y = int(int(idx / int(self.width*1.75)) / 1.75)
        self.SRRFArray[idx] = self.pixels[y * self.width + x]
        return

    def return_result(self):
        return self.SRRFArray

                    
                
