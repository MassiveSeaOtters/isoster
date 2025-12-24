from typing import Optional
from pydantic import BaseModel, Field, model_validator

class IsosterConfig(BaseModel):
    """
    Configuration for isoster fitting.
    
    Provides type safety and validation for fitting parameters.
    """
    # Geometry initialization
    x0: Optional[float] = Field(None, description="Initial center x coordinate. If None, uses image center.")
    y0: Optional[float] = Field(None, description="Initial center y coordinate. If None, uses image center.")
    eps: float = Field(0.2, ge=0.0, lt=1.0, description="Initial ellipticity.")
    pa: float = Field(0.0, description="Initial position angle in radians.")
    
    # SMA control
    sma0: float = Field(10.0, gt=0.0, description="Starting semi-major axis length.")
    minsma: float = Field(0.0, ge=0.0, description="Minimum SMA to fit.")
    maxsma: Optional[float] = Field(None, gt=0.0, description="Maximum SMA to fit. If None, uses image size/2.")
    astep: float = Field(0.1, gt=0.0, description="Step size for SMA growth.")
    linear_growth: bool = Field(False, description="If True, sma = sma + astep. Else sma = sma * (1 + astep).")
    
    # Fitting control
    maxit: int = Field(50, gt=0, description="Maximum iterations per isophote.")
    minit: int = Field(10, gt=0, description="Minimum iterations per isophote.")
    conver: float = Field(0.05, gt=0.0, description="Convergence threshold (max harmonic amplitude / rms).")
    
    # Quality control
    sclip: float = Field(3.0, gt=0.0, description="Sigma clipping threshold.")
    nclip: int = Field(0, ge=0, description="Number of sigma clipping iterations.")
    sclip_low: Optional[float] = Field(None, description="Lower sigma clipping threshold.")
    sclip_high: Optional[float] = Field(None, description="Upper sigma clipping threshold.")
    fflag: float = Field(0.5, ge=0.0, le=1.0, description="Maximum fraction of flagged data (masked + clipped).")
    maxgerr: float = Field(0.5, gt=0.0, description="Maximum relative error in local radial gradient.")
    
    # Constraints
    fix_center: bool = Field(False, description="Fix center coordinates during fitting.")
    fix_pa: bool = Field(False, description="Fix position angle during fitting.")
    fix_eps: bool = Field(False, description="Fix ellipticity during fitting.")
    
    # Outputs & Features
    compute_errors: bool = Field(True, description="Calculate parameter errors.")
    compute_deviations: bool = Field(True, description="Calculate higher-order harmonic deviations (a3, b3, etc.).")
    full_photometry: bool = Field(False, description="Calculate flux integration metrics (tflux_e, etc.).")
    debug: bool = Field(False, description="Include debug info in results and enable verbose calculation.")

    # Integration Mode
    integrator: str = Field(default='mean', pattern='^(mean|median)$', description="Integration method for flux calculation.")

    @model_validator(mode='after')
    def check_sma_consistency(self):
        if self.maxsma is not None and self.maxsma < self.minsma:
            raise ValueError(f"maxsma ({self.maxsma}) must be greater than minsma ({self.minsma})")
        return self
