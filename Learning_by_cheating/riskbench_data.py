import numpy as np 
from typing import NamedTuple, List, Tuple, Optional
import cv2 
import os
import ujson

from mask import PixelDimensions, square_fitting_rect_at_any_rotation, MapMaskGenerator, RenderingWindow, BirdViewMasks, Coord, Loc, COLOR_OFF, COLOR_ON

RgbCanvas = np.ndarray  # [np.uint8] with shape (y, x, 3)
BirdView = np.ndarray  # [np.uint8] with shape (level, y, x)


class RGB:
    VIOLET = (173, 127, 168)
    ORANGE = (252, 175, 62)
    CHOCOLATE = (233, 185, 110)
    CHAMELEON = (138, 226, 52)
    SKY_BLUE = (114, 159, 207)
    DIM_GRAY = (105, 105, 105)
    DARK_GRAY = (50, 50, 50)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    WHITE = (255, 255, 255)
    UNLABELES = (0, 0, 0)

# 9 channel 
RGB_BY_MASK = {



    BirdViewMasks.AGENT: RGB.CHAMELEON,
    BirdViewMasks.OBSTACLES: RGB.DARK_GRAY,    
    BirdViewMasks.PEDESTRIANS: RGB.CHOCOLATE,
    BirdViewMasks.VEHICLES: RGB.ORANGE,
    BirdViewMasks.ROAD_LINE: RGB.WHITE,
    BirdViewMasks.ROAD: RGB.DIM_GRAY,
    
    BirdViewMasks.UNLABELES: RGB.UNLABELES,
    
    
    # 
    
}




class CroppingRect(NamedTuple):
    x: int
    y: int
    width: int
    height: int

    @property
    def vslice(self) -> slice:
        return slice(self.y, self.y + self.height)

    @property
    def hslice(self) -> slice:
        return slice(self.x, self.x + self.width)


def rotate(image, angle, center=None, scale=1.0):
    assert image.dtype == np.uint8

    """Copy paste of imutils method but with INTER_NEAREST and BORDER_CONSTANT flags"""
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # return the rotated image
    return rotated



class BirdViewProducer:
    
    def __init__(self,
                 Town_Name,
                 target_size: PixelDimensions,
                 pixels_per_meter: int = 4
                 ) -> None:
        
        
        self.target_size = target_size
        self._pixels_per_meter = pixels_per_meter
        
        
        rendering_square_size = round(square_fitting_rect_at_any_rotation(self.target_size))
        self.rendering_area = PixelDimensions(
            width=rendering_square_size, height=rendering_square_size
        )
        # self._crop_type = crop_type
        
        
    
        self.masks_generator = MapMaskGenerator(town=Town_Name, pixels_per_meter=self._pixels_per_meter)
        
        
        # self.full_road_cache = self.masks_generator.road_mask()
        # self.full_lanes_cache = self.masks_generator.road_line_mask()
        
        # self.road_mask()
        # self.road_line_mask()
        # self.crosswalk_mask()
        
        
    # draw 
    def produce(
        self, vehicle_loc, yaw, agent_bbox_list, vehicle_bbox_list, pedestrians_bbox_list, obstacle_bbox_list, 
    ) -> BirdView:
        

        # Reusing already generated static masks for whole map
        self.masks_generator.disable_local_rendering_mode()
        agent_global_px_pos = self.masks_generator.location_to_pixel(vehicle_loc)
        # agent_global_px_pos:  Coord(x=461, y=760) 

        cropping_rect = CroppingRect(
            x=int(agent_global_px_pos.x - self.rendering_area.width / 2),
            y=int(agent_global_px_pos.y - self.rendering_area.height / 2),
            width=self.rendering_area.width,
            height=self.rendering_area.height,
        )

        
        # CroppingRect(x=333, y=632, width=255, height=255)
        
        masks = np.zeros(
            shape=(
                len(BirdViewMasks),
                self.rendering_area.height,
                self.rendering_area.width,
            ),
            dtype=np.uint8,
        )
        
        # masks[BirdViewMasks.ROAD.value] = self.full_road_cache[
        #     cropping_rect.vslice, cropping_rect.hslice
        # ]
        # masks[BirdViewMasks.ROAD_LINE.value] = self.full_lanes_cache[
        #     cropping_rect.vslice, cropping_rect.hslice
        # ]
        
        rendering_window = RenderingWindow(
            origin=vehicle_loc, area=self.rendering_area
        )
        self.masks_generator.enable_local_rendering_mode(rendering_window)
        
        
        # draw actor 
        masks = self._render_actors_masks(agent_bbox_list, 
                                          vehicle_bbox_list, 
                                          pedestrians_bbox_list, 
                                          obstacle_bbox_list, 
                                          masks)
        
        #         agent_bbox_list,
        # vehicle_bbox_list,
        # pedestrians_bbox_list,
        # obstacle_bbox_list,
        
        
        # -------------------------------------------------------------------- # 
        
        cropped_masks = self.apply_agent_following_transformation_to_masks(
           yaw, masks,
        )
        ordered_indices = [mask.value for mask in BirdViewMasks.bottom_to_top()]

        
        return cropped_masks[ordered_indices]
    
    

    def _render_actors_masks(
        self,
        # agent_vehicle: carla.Actor,
        # segregated_actors: SegregatedActors,
        agent_bbox_list,
        vehicle_bbox_list,
        pedestrians_bbox_list,
        obstacle_bbox_list,
        masks: np.ndarray,
    ) -> np.ndarray:
        """Fill masks with ones and zeros (more precisely called as "bitmask").
        Although numpy dtype is still the same, additional semantic meaning is being added.
        """
        # lights_masks = self.masks_generator.traffic_lights_masks(
        #     segregated_actors.traffic_lights
        # )
        
        # red_lights_mask, yellow_lights_mask, green_lights_mask = lights_masks
        
        
        
        # masks[BirdViewMasks.RED_LIGHTS.value] = red_lights_mask
        # masks[BirdViewMasks.YELLOW_LIGHTS.value] = yellow_lights_mask
        # masks[BirdViewMasks.GREEN_LIGHTS.value] = green_lights_mask
        
        masks[BirdViewMasks.AGENT.value] = self.masks_generator.draw_bbox_mask(
            agent_bbox_list
        )
        masks[BirdViewMasks.VEHICLES.value] = self.masks_generator.draw_bbox_mask(
            vehicle_bbox_list
        )
        masks[BirdViewMasks.PEDESTRIANS.value] = self.masks_generator.draw_bbox_mask(
            pedestrians_bbox_list
        )
        masks[BirdViewMasks.OBSTACLES.value] = self.masks_generator.draw_bbox_mask(
            obstacle_bbox_list
        )
        
        
        return masks
      
    @staticmethod
    def as_rgb(birdview: BirdView) -> RgbCanvas:
        _, h, w = birdview.shape
        rgb_canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        nonzero_indices = lambda arr: arr == COLOR_ON

        for mask_type in BirdViewMasks.bottom_to_top():
            
            # print(mask_type)
            
            rgb_color = RGB_BY_MASK[mask_type]
            # print(rgb_color)
            mask = birdview[mask_type]
            # If mask above contains 0, don't overwrite content of canvas (0 indicates transparency)
            rgb_canvas[nonzero_indices(mask)] = rgb_color
        return rgb_canvas
    
    
    @staticmethod
    def as_ss(birdview: BirdView):
        _, h, w = birdview.shape
        canvas = np.zeros(shape=(h, w), dtype=np.uint8)
        nonzero_indices = lambda arr: arr == COLOR_ON

        for mask_type in BirdViewMasks.bottom_to_top():
            # print(mask_type)
            # print( mask_type.value)
            mask = birdview[mask_type]
            # If mask above contains 0, don't overwrite content of canvas (0 indicates transparency)
            canvas[nonzero_indices(mask)] = mask_type.value
        return canvas
    
    def apply_agent_following_transformation_to_masks(
        self, yaw,  masks: np.ndarray,
    ) -> np.ndarray:
        
        # agent_transform = agent_vehicle.get_transform()
        angle = ( yaw + 90)  # vehicle's front will point to the top

        # Rotating around the center
        crop_with_car_in_the_center = masks
        masks_n, h, w = crop_with_car_in_the_center.shape
        rotation_center = Coord(x=w // 2, y=h // 2)

        # warpAffine from OpenCV requires the first two dimensions to be in order: height, width, channels
        crop_with_centered_car = np.transpose(
            crop_with_car_in_the_center, axes=(1, 2, 0)
        )
        rotated = rotate(crop_with_centered_car, angle, center=rotation_center)
        rotated = np.transpose(rotated, axes=(2, 0, 1))

        half_width = self.target_size.width // 2
        hslice = slice(rotation_center.x - half_width, rotation_center.x + half_width)

        # if self._crop_type is BirdViewCropType.FRONT_AREA_ONLY:
        #     vslice = slice(rotation_center.y - self.target_size.height, rotation_center.y)
        # elif self._crop_type is BirdViewCropType.FRONT_AND_REAR_AREA:
            
        half_height = self.target_size.height // 2
        vslice = slice(
            rotation_center.y - half_height, rotation_center.y + half_height
        )
        
            
        # else:
        #     raise NotImplementedError
        assert (
            vslice.start > 0 and hslice.start > 0
        ), "Trying to access negative indexes is not allowed, check for calculation errors!"
        car_on_the_bottom = rotated[:, vslice, hslice]
        return car_on_the_bottom

        
       