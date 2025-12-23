import numpy as np
from astropy.table import Table
from astropy.io import fits

def isophote_results_to_astropy_tables(results):
    """
    Convert isophote results to an Astropy Table.
    
    Parameters
    ----------
    results : dict
        The dictionary returned by fit_image(), containing 'isophotes' and 'config'.
        
    Returns
    -------
    table : astropy.table.Table
        Astropy Table containing the isophote data with standard columns:
        - sma: Semi-major axis length
        - intens: Mean intensity
        - intens_err: Intensity error
        - eps: Ellipticity
        - pa: Position angle (radians)
        - x0, y0: Center coordinates
        - stop_code: Fitting status code
        - niter: Number of iterations
    """
    # Handle case where results is just the list (backward compatibility or direct list usage)
    if isinstance(results, list):
        isophotes = results
    else:
        isophotes = results.get('isophotes', [])
    
    if not isophotes:
        return Table()
        
    # Create table from list of dicts
    tbl = Table(rows=isophotes)
    
    # Reorder columns for better readability
    # Common columns first
    common_cols = ['sma', 'intens', 'intens_err', 'eps', 'pa', 'x0', 'y0', 'rms', 'stop_code', 'niter']
    
    # Get all columns present
    all_cols = tbl.colnames
    
    # Create new order
    new_order = [c for c in common_cols if c in all_cols] + [c for c in all_cols if c not in common_cols]
    
    tbl = tbl[new_order]
    
    return tbl

def isophote_results_to_fits(results, filename, overwrite=True):
    """
    Save isophote results to a FITS table, including configuration parameters as header keywords.
    
    Parameters
    ----------
    results : dict
        The dictionary returned by fit_image().
    filename : str
        Output filename.
    overwrite : bool
        Whether to overwrite existing file.
    """
    tbl = isophote_results_to_astropy_tables(results)
    
    # Add config to header
    # We can add them to the table's meta, which writes to the primary header or table header
    if isinstance(results, dict):
        config = results.get('config', {})
        
        for key, value in config.items():
            # FITS keywords should be short. We'll use the key as is, astropy handles it.
            # We filter out None values as they can't be written to FITS header easily
            if value is None:
                continue
                
            # Check value type
            if isinstance(value, (str, int, float, bool, np.number)):
                tbl.meta[key] = value
            else:
                # Convert to string if complex (e.g. list)
                tbl.meta[key] = str(value)
            
    tbl.write(filename, format='fits', overwrite=overwrite)
