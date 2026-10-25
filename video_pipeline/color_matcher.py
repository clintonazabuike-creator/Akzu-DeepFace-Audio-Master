import cv2
import numpy as np

def match_colors_lab(source_face: np.ndarray, target_region: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Performs color transfer in CIELAB color space from source_face to match target_region.
    Maintains structural details while aligning skin tones and illumination gradients.
    
    Args:
        source_face (np.ndarray): BGR image of the swapped face.
        target_region (np.ndarray): BGR image of the original destination face/skin region.
        mask (np.ndarray, optional): Alpha/feather mask for seamless blending.
        
    Returns:
        np.ndarray: Color-harmonized BGR image.
    """
    # Convert images to float32 for precise channel arithmetic
    src_lab = cv2.cvtColor(source_face, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    matched_lab = np.empty_like(src_lab)
    
    # Process L, A, B channels independently (Mean & Standard Deviation transfer)
    for i in range(3):
        src_channel = src_lab[:, :, i]
        tgt_channel = tgt_lab[:, :, i]
        
        src_mean = np.mean(src_channel)
        src_std = np.std(src_channel)
        tgt_mean = np.mean(tgt_channel)
        tgt_std = np.std(tgt_channel)
        
        # Prevent division by zero on uniform channels
        if src_std < 1e-5:
            src_std = 1.0
            
        # Scale and shift distribution
        normalized = (src_channel - src_mean) * (tgt_std / src_std) + tgt_mean
        matched_lab[:, :, i] = np.clip(normalized, 0.0, 255.0)
        
    # Convert back to BGR space
    matched_bgr = cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # Apply feathered mask blending if provided
    if mask is not None:
        if len(mask.shape) == 2:
            m = mask.astype(np.float32) / 255.0
            m = np.expand_dims(m, axis=2)
        else:
            m = mask.astype(np.float32) / 255.0
            
        if m.shape[2] == 1:
            m = np.repeat(m, 3, axis=2)
            
        blended = (matched_bgr * m + source_face * (1.0 - m)).astype(np.uint8)
        return blended
        
    return matched_bgr
