import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img

def load_and_align_atlas(atlas_nifti_path, reference_nifti_path=None):
    """
    加載圖譜文件。如果提供了參考圖譜路徑，且兩者維度或坐標系不一致，
    則自動將當前圖譜重採樣（Resample）到參考空間。
    
    參數:
        atlas_nifti_path (str): 當前要處理的圖譜路徑。
        reference_nifti_path (str, 可選): 參考圖譜路徑（通常選定第一個處理的圖譜）。
        
    返回:
        atlas_img (nibabel.Nifti1Image): 對齊後的 NIfTI 圖像對象。
    """
    atlas_img = nib.load(atlas_nifti_path)
    
    if reference_nifti_path is not None:
        ref_img = nib.load(reference_nifti_path)
        
        # 檢查圖像的形狀 (Shape) 和仿射矩陣 (Affine) 是否完全一致
        shape_mismatch = atlas_img.shape != ref_img.shape
        affine_mismatch = not np.allclose(atlas_img.affine, ref_img.affine)
        
        if shape_mismatch or affine_mismatch:
            # 【關鍵】：圖譜包含的是離散的腦區標籤（整數），必須使用 nearest 插值，
            # 否則會產生 1.5 號腦區這種毫無意義的數值。
            atlas_img = resample_to_img(
                source_img=atlas_img, 
                target_img=ref_img, 
                interpolation="nearest"
            )
            
    return atlas_img

def project_to_voxel_space(nodal_strengths, atlas_img):
    """
    第一步：空間概率貢獻地圖投影 (Voxel-wise Projection)
    將一維的腦區節點得分，映射回三維標準腦空間中對應的體素上。
    
    參數:
        nodal_strengths (np.ndarray): 形狀為 (N,) 的一維數組，對應 N 個腦區的得分。
        atlas_img (nibabel.Nifti1Image): 已經加載（並可能已經對齊）的圖譜對象。
        
    返回:
        volume_data (np.ndarray): 映射後的三維體素矩陣。
        affine (np.ndarray): 該空間的仿射矩陣。
    """
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine
    
    labels = np.unique(atlas_data)
    labels = labels[labels != 0]
    
    if len(nodal_strengths) != len(labels):
        raise ValueError(f"維度不匹配: 節點得分數量({len(nodal_strengths)}) 與 圖譜實際ROI標籤數量({len(labels)}) 不一致！")
    
    volume_data = np.zeros_like(atlas_data, dtype=np.float32)
    
    for idx, label in enumerate(labels):
        score = nodal_strengths[idx]
        volume_data[atlas_data == label] = score
        
    return volume_data, affine

def calculate_gci(volume_list):
    """
    第二步：全局一致性指數計算 (Global Consistency Index, GCI)
    """
    z_scored_volumes = []
    
    for volume in volume_list:
        brain_mask = volume != 0
        
        if not np.any(brain_mask):
            z_volume = np.zeros_like(volume)
        else:
            mean_val = np.mean(volume[brain_mask])
            std_val = np.std(volume[brain_mask])
            z_volume = np.zeros_like(volume, dtype=np.float32)
            z_volume[brain_mask] = (volume[brain_mask] - mean_val) / (std_val + 1e-8)
            
        z_scored_volumes.append(z_volume)
    
    z_scored_volumes = np.array(z_scored_volumes)
    gci_volume = np.mean(z_scored_volumes, axis=0)
    
    return gci_volume