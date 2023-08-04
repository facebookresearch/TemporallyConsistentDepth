import os
import sys
import torch
import cv2
from torchvision.transforms import Compose
sys.path.append( os.path.join(os.getcwd(), 'DPT'))
from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

# Source code adapted from https://github.com/isl-org/DPT
class DPTWrapper:

    def __init__(self, model_path, model_type="dpt_hybrid", optimize=True):
        super(DPTWrapper, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == "dpt_large":  # DPT-Large
            net_w = net_h = 384
            model = DPTDepthModel(
                path=model_path,
                backbone="vitl16_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_type == "dpt_hybrid":  # DPT-Hybrid
            net_w = net_h = 384
            model = DPTDepthModel(
                path=model_path,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_type == "dpt_hybrid_kitti":
            net_w, net_h = 1216, 352
            model = DPTDepthModel(
                path=model_path,
                scale=0.00006016,
                shift=0.00579,
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_type == "dpt_hybrid_nyu":
            net_w, net_h = 640, 480
            model = DPTDepthModel(
                path=model_path,
                scale=0.000305,
                shift=0.1378,
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_type == "midas_v21":  # Convolutional model
            net_w = net_h = 384
            model = MidasNet_large(model_path, non_negative=True)
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            assert (
                False
            ), f"model_type '{model_type}' not implemented."

        self.transform = Compose([
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet()]
        )
        self.model = model
        self.net_w = net_w
        self.net_h = net_h
        self.normalization = normalization
        self.model.eval()
        self.optimize = optimize
        self.model_type = model_type
        
        # set torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        if optimize == True and self.device == torch.device("cuda"):
            self.model = self.model.to(memory_format=torch.channels_last)
            self.model = self.model.half()
        self.model.to(self.device)

        
    def __call__( self, img ):
        """ 
        Estimate monocular depth for the given RGB input image.
        The image values should be in the range [0, 1].
        """
        img_input = self.transform({"image": img})["image"]
        
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)

            if self.optimize == True and self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            
            if self.model_type == "dpt_hybrid_kitti":
                prediction *= 256

            if self.model_type == "dpt_hybrid_nyu":
                prediction *= 1000.0

        return prediction 


