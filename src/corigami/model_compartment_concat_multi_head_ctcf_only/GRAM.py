import time
import torch
import torch.nn.functional as F
 
 
 
class GradCAM:
    # 初始化，得到target_layer层
    def __init__(self, model, layer_name, img_size=(640, 640)):
        self.model = model
        self.gradients = dict()
        self.activations = dict()
 
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
 
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None
 
        
        # 获取forward过程中每层的输入和输出，用于对比hook是不是正确记录
        model.decoder.conv_end.register_forward_hook(forward_hook)
        model.decoder.conv_end.register_full_backward_hook(backward_hook)
        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device))
 
    def forward(self, input_img, class_idx=True):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_maps = []
        b, c, h, w = input_img.size()
        preds, logits = self.model(input_img)
        for logit, cls, cls_name in zip(logits[0], preds[1][0], preds[2][0]):
            if class_idx:
                score = logit[cls]
            else:
                score = logit.max()
            self.model.zero_grad()
            tic = time.time()
            # 获取梯度
            score.backward(retain_graph=True)
            print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')
            gradients = self.gradients['value']
            activations = self.activations['value']
            b, k, u, v = gradients.size()
            alpha = gradients.view(b, k, -1).mean(2)
            weights = alpha.view(b, k, 1, 1)
            saliency_map = (weights * activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)
            saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
            saliency_maps.append(saliency_map)
        return saliency_maps, logits, preds
 
    def __call__(self, input_img):
        return self.forward(input_img)
 

def main():
    model = ModelClass(num_genomic_features, mid_hidden = mid_hidden)
    # img[..., ::-1]: BGR --> RGB
    # (480, 640, 3) --> (1, 3, 480, 640)
    torch_img = model.preprocessing(img[..., ::-1])
    tic = time.time()
    # 遍历三层检测层
    for target_layer in target_layers:
        # 获取grad-cam方法
        if args.method == 'gradcam':
            saliency_method = YOLOV5GradCAM(model=model, layer_name=target_layer, img_size=input_size)
        elif args.method == 'gradcampp':
            saliency_method = YOLOV5GradCAMPP(model=model, layer_name=target_layer, img_size=input_size)
        masks, logits, [boxes, _, class_names, conf] = saliency_method(torch_img)  # 得到预测结果
        result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
        result = result[..., ::-1]