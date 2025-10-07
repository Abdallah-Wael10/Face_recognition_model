# recover_encodings.py
import pickle
import numpy as np
import os

def recover_encodings():
    input_path = r"D:\Bureau_Tutorials\Hikvision-Face-Recognition\model\encodings.pickle"
    output_path = r"D:\Bureau_Tutorials\Hikvision-Face-Recognition\model\encodings_fixed.pickle"
    
    try:
        # Try to load with error handling
        with open(input_path, 'rb') as f:
            # Use a custom unpickler to handle the numpy core issue
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == "numpy.core" and name == "multiarray":
                        return np.core.multiarray
                    if module == "numpy.core" and name == "_reconstruct":
                        return np.core._reconstruct
                    if module == "numpy" and name == "ndarray":
                        return np.ndarray
                    if module == "numpy" and name == "dtype":
                        return np.dtype
                    return super().find_class(module, name)
            
            data = CustomUnpickler(f).load()
        
        print("✅ Successfully recovered encodings!")
        print(f"Data type: {type(data)}")
        
        # Save with current numpy version
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Saved fixed encodings to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Recovery failed: {e}")
        return False

if __name__ == "__main__":
    recover_encodings()