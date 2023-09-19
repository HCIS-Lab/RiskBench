import numpy as np
import json
import os 
from typing import NamedTuple, List, Tuple, Optional
from enum import IntEnum, auto, Enum
from PIL import Image
import cv2 




Mask = np.ndarray  # of shape (y, x), stores 0 and 1, dtype=np.int32
Meters = float
MAP_BOUNDARY_MARGIN: Meters = 200 

# DEFAULT_HEIGHT = 336  # its 84m when density is 4px/m
# DEFAULT_WIDTH = 150  # its 37.5m when density is 4px/m

COLOR_OFF = 0
COLOR_ON = 1

class Loc(NamedTuple):
    x: float
    y: float

class Coord(NamedTuple):
    x: int
    y: int
    
class MapBoundaries(NamedTuple):
    """Distances in carla.World coordinates"""

    min_x: Meters
    min_y: Meters
    max_x: Meters
    max_y: Meters
    
    
class Dimensions(NamedTuple):
    width: int
    height: int

PixelDimensions = Dimensions
loc = Loc

class RenderingWindow(NamedTuple):
    origin: loc
    area: PixelDimensions
    



PixelDimensions = Dimensions





class MapMaskGenerator:
    
    def __init__(self, town, pixels_per_meter: int = 4) -> None:
        
        
        self.Town = town
        self.pixels_per_meter = pixels_per_meter
        self.rendering_window: Optional[RenderingWindow] = None
        
        

        self._map_boundaries = self._find_map_boundaries()
        self._mask_size: PixelDimensions = self.calculate_mask_size()
        
        

        
        
    def disable_local_rendering_mode(self):
        self.rendering_window = None

    def enable_local_rendering_mode(self, rendering_window: RenderingWindow):
        self.rendering_window = rendering_window
        
        
    def _find_map_boundaries(self) -> MapBoundaries:
        """Find extreme locations on a map.
        It adds a decent margin because waypoints lie on the road, which means
        that anything that is slightly further than the boundary
        could cause out-of-range exceptions (e.g. pavements, walkers, etc.)
        
        # road = (n, 3)      
        """   
        Generate_MapBoundaries = False

        if os.path.isfile("./MapBoundaries.json"):
            
            with open("./MapBoundaries.json", 'r') as f:
                data = json.load(f)
            try:
                min_x= data[self.Town]["min_x"]
                min_y= data[self.Town]["min_y"]
                max_x= data[self.Town]["max_x"]
                max_y= data[self.Town]["max_y"]
            except:
                Generate_MapBoundaries = True
        else:
            Generate_MapBoundaries = True
        
        if Generate_MapBoundaries:
            if os.path.isfile("./MapBoundaries.json"):
                with open("./MapBoundaries.json", 'r') as f:
                    data = json.load(f)
            else:
                data = {}
            self._road_waypoints = np.load(f"./maps/{self.Town}/mask_road.npy")
            min_x= min( self._road_waypoints, key=lambda x: x[0] )[0] - MAP_BOUNDARY_MARGIN
            min_y= min( self._road_waypoints, key=lambda x: x[1] )[1] - MAP_BOUNDARY_MARGIN
            max_x= max( self._road_waypoints, key=lambda x: x[0] )[0] + MAP_BOUNDARY_MARGIN
            max_y= max( self._road_waypoints, key=lambda x: x[1] )[1] + MAP_BOUNDARY_MARGIN
            data[self.Town] = {}
            data[self.Town]["min_x"] = min_x
            data[self.Town]["min_y"] = min_y
            data[self.Town]["max_x"] = max_x
            data[self.Town]["max_y"] = max_y
            
            f = open("./MapBoundaries.json", "w")
            json.dump(data, f, indent=4)
                    
        return MapBoundaries(
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
        )
        
    def calculate_mask_size(self) -> PixelDimensions:
        """Convert map boundaries to pixel resolution."""
        width_in_meters = self._map_boundaries.max_x - self._map_boundaries.min_x
        height_in_meters = self._map_boundaries.max_y - self._map_boundaries.min_y
        width_in_pixels = int(width_in_meters * self.pixels_per_meter)
        height_in_pixels = int(height_in_meters * self.pixels_per_meter)
        
        return PixelDimensions(width=width_in_pixels, height=height_in_pixels)
        
    def make_empty_mask(self) -> Mask:
        if self.rendering_window is None:
            shape = (self._mask_size.height, self._mask_size.width)
        else:
            shape = (
                self.rendering_window.area.height,
                self.rendering_window.area.width,
            )
        return np.zeros(shape, np.uint8)
    

        
        # return  

    # def pedestrians_mask(self, peds_bbox_list) -> Mask:
    #     canvas = self.make_empty_mask()
    #     # Carla to pixel 
        
    #     for corners in peds_bbox_list:
    #         corners = [self.location_to_pixel(loc) for loc in corners]
    #         cv2.fillPoly(img=canvas, pts=np.int32([corners]), color=COLOR_ON)
    #     return canvas    
        
    def draw_bbox_mask(self, bbox_list) -> Mask:
        canvas = self.make_empty_mask()
        # Carla to pixel 
        
        for corners in bbox_list:
            corners = [self.location_to_pixel(loc) for loc in corners]
            cv2.fillPoly(img=canvas, pts=np.int32([corners]), color=COLOR_ON)
        return canvas    
        
        
    def crosswalk_mask(self) -> Mask:

        min_x = self._map_boundaries.min_x
        min_y = self._map_boundaries.min_y
        min_point = np.array([min_x, min_y])

        canvas = self.make_empty_mask()
        
        if os.path.exists(f"./maps/{self.Town}/crosswalk.npy"):
            road = np.load(f"./maps/{self.Town}/crosswalk.npy")
            road_pixel = np.rint(self.pixels_per_meter * (road - min_point))
            road_pixel = road_pixel.astype(int)[:,:2]
            canvas[road_pixel[:,1], road_pixel[:,0]] = COLOR_ON
          
        # For debug   
        # img = 255 * canvas
        # img = img.astype(np.uint8)
        # Image.fromarray(img).save('mask_crosswalk.png')
        
        return canvas
    
    def road_mask(self) -> Mask:
        canvas = self.make_empty_mask()
            
        
        
        self._road_waypoints = np.load(f"./maps/{self.Town}/mask_road.npy")
        road = self._road_waypoints
        min_x = self._map_boundaries.min_x
        min_y = self._map_boundaries.min_y
        min_point = np.array([min_x, min_y])
        
        # # Pixel coordinates on full map
        # x = int(self.pixels_per_meter * (loc.x - min_x))
        # y = int(self.pixels_per_meter * (loc.y - min_y))
        
        road_pixel = np.rint(self.pixels_per_meter * (road - min_point))
        
        
        road_pixel = road_pixel.astype(int)[:,:2]
        
        canvas[road_pixel[:,1], road_pixel[:,0]] = COLOR_ON
        
        
        # For debug  
        # img = 255 * canvas
        # img = img.astype(np.uint8)
        # Image.fromarray(img).save('mask_lane.png')
        
        return canvas
    
    def road_line_mask(self) -> Mask:
        canvas = self.make_empty_mask()
        
        # road = (34514084, 3) ( x, y, z)
        # canvas.shape (3270, 3328)
        
        # self._road_line_waypoints 
        
        road = np.load(f"./maps/{self.Town}/mask_road_line.npy")
        
        min_x = self._map_boundaries.min_x
        min_y = self._map_boundaries.min_y
        min_point = np.array([min_x, min_y])
        
        # # Pixel coordinates on full map
        # x = int(self.pixels_per_meter * (loc.x - min_x))
        # y = int(self.pixels_per_meter * (loc.y - min_y))
        
        road_pixel = np.rint(self.pixels_per_meter*(road - min_point))
        road_pixel = road_pixel.astype(int)[:,:2]
        

        
        canvas[road_pixel[:,1], road_pixel[:,0]] = COLOR_ON
        

        # img = 255 * canvas
        # img = img.astype(np.uint8)
        # Image.fromarray(img).save('mask_road_line.png')


        return canvas
        
    def location_to_pixel(self, loc) -> Coord:
        """Convert world coordinates to pixel coordinates.

        For example: top leftmost location will be a pixel at (0, 0).
        """
        min_x = self._map_boundaries.min_x
        min_y = self._map_boundaries.min_y
        
        # Pixel coordinates on full map
        x = int(self.pixels_per_meter * (loc.x - min_x))
        y = int(self.pixels_per_meter * (loc.y - min_y))
        
        if self.rendering_window is not None:
            # global rendering area coordinates
            origin_x = self.pixels_per_meter * (self.rendering_window.origin.x - min_x)
            origin_y = self.pixels_per_meter * (self.rendering_window.origin.y - min_y)
            topleft_x = int(origin_x - self.rendering_window.area.width / 2)
            topleft_y = int(origin_y - self.rendering_window.area.height / 2)

            # x, y becomes local coordinates within rendering window
            x -= topleft_x
            y -= topleft_y

        return Coord(x=int(x), y=int(y))


def circle_circumscribed_around_rectangle(rect_size: Dimensions) -> float:
    """Returns radius of that circle."""
    a = rect_size.width / 2
    b = rect_size.height / 2
    return float(np.sqrt(np.power(a, 2) + np.power(b, 2)))

def square_fitting_rect_at_any_rotation(rect_size: Dimensions) -> float:
    """Preview: https://pasteboard.co/J1XK62H.png"""
    

    radius = circle_circumscribed_around_rectangle(rect_size)    
    side_length_of_square_circumscribed_around_circle = radius * 2
    return side_length_of_square_circumscribed_around_circle

class BirdViewMasks(IntEnum):
    # 
    # RED_LIGHTS = 7
    # YELLOW_LIGHTS = 6
    # GREEN_LIGHTS = 5
    
    AGENT = 6
    OBSTACLES = 5
    PEDESTRIANS = 4
    VEHICLES = 3
    # crosswalk
    ROAD_LINE = 2
    ROAD = 1
    UNLABELES = 0

    @staticmethod
    def top_to_bottom() -> List[int]:
        return list(BirdViewMasks)

    @staticmethod
    def bottom_to_top() -> List[int]:
        return list(reversed(BirdViewMasks.top_to_bottom()))
