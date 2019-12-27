import os

os.makedirs("models", exist_ok=True)
#torch.save(model, "models/model.pt")
os.makedirs("models/state_dict", exist_ok=True)
#os.rmdir("folderName")

import glob, os
for i in glob.glob("models/state_dict/*"):
   os.remove(i)

