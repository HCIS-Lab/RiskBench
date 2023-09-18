from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor

from detectron2.config import get_cfg
from detectron2.structures.image_list import ImageList
import torch
from torchvision import transforms
import cv2
from PIL import Image

def get_config():
	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
	# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
	path = cfg.MODEL.WEIGHTS
	# model = build_model(cfg)
	# DetectionCheckpointer(model).load(path)
	# return model, cfg
	return cfg

# def get_maskrcnn():
# 	cfg = get_cfg()

# 	# model.eval()

# 	predictor = Predictor(cfg)
# 	# features, proposals, instances = predictor(original_image)

def get_maskrcnn(device):
	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
	path = cfg.MODEL.WEIGHTS
	if device == "cpu":
		cfg.MODEL.DEVICE = device
	model = build_model(cfg)
	return model

class Predictor:
	def __init__(self, cfg):
		self.cfg = cfg.clone()  # cfg can be modified by model
		self.model = build_model(self.cfg)
		self.model.eval()
		if len(cfg.DATASETS.TEST):
		    self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

		checkpointer = DetectionCheckpointer(self.model)
		checkpointer.load(cfg.MODEL.WEIGHTS)

		self.aug = T.ResizeShortestEdge(
		    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
		)

		self.input_format = cfg.INPUT.FORMAT
		assert self.input_format in ["RGB", "BGR"], self.input_format

	def __call__(self, original_image):
		with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
			if self.input_format == "RGB":
            	# whether the model expects BGR inputs or RGB
				original_image = original_image[:, :, ::-1]
			height, width = original_image.shape[:2]
			image = self.aug.get_transform(original_image).apply_image(original_image)
			image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

			inputs = {"image": image, "height": height, "width": width}
			features = self.model.backbone([inputs])[0]
			proposals, _ = self.model.proposal_generator(image, features)
			instances, _ = self.model.roi_heads(image, features, proposals)
			return features, proposals, instances
# # test

# # loader = transforms.Compose([
# #     transforms.ToTensor()])

# # # im = cv2.imread("1.jpg")
# # im = Image.open('1.jpg').convert('RGB')
# # # im = Image.fromarray(im)
# # img = loader(im).unsqueeze(0)
# # im = img.to('cuda', torch.float)
# # print(im.shape)
# # im = ImageList.from_tensors([im.cuda()])
# # tensor = torch.ones([1, 3, 224, 224], dtype=torch.float32).cuda()
# cfg = get_maskrcnn()
# # model = model.cuda()
# # print(model.backbone.__dict__)
# original_image = cv2.imread("1.jpg")
# print(type(original_image))
# # with torch.no_grad():
# #     # original_image = original_image[:, :, ::-1]
# #     original_image = original_image[:, :225, :225]
# #     height, width = original_image.shape[:2]
# #     image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
# #     image = image.view(1, 3, height, width)
# #     image = image.cuda()
# # model.eval()

# predictor = Predictor(cfg)
# features, proposals, instances = predictor(original_image)



# ResNet features
# features = model.backbone.bottom_up(tensor)

# features = model.backbone(image)

# print(features['res5'].shape)
# for k, v in features.items():
# 	print(k)

# print('---------------------------------')
# FPN features
# features = model.backbone(tensor)
# for k, v in features.items():
# 	print(k)

# proposals, _ = model.proposal_generator(original_image, features)
# instances, _ = model.roi_heads(original_image, features, proposals)
