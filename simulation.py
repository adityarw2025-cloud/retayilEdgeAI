# simulation.py
"""
Retail Store Customer Simulation Engine.
Generates synthetic video frames of customers moving through Store Area,
Shelf Area, and Checkout Area to demonstrate computer-vision analytics
(customer count, queue count, dwell time, and zone occupancy) without a webcam.
"""
import os
import math
import random
import time
import cv2
import numpy as np


class SimulatedCustomer:
    """Represents a simulated shopper navigating retail store zones."""
    def __init__(self, customer_id: int, sprite_idx: int, path_type: str = "standard", speed_factor: float = 1.0):
        self.id = customer_id
        self.sprite_idx = sprite_idx
        self.path_type = path_type
        self.speed_factor = speed_factor

        # Waypoints: [(x, y, dwell_seconds_at_target)]
        self.waypoints = []
        self._build_trajectory()
        self.current_wp_idx = 0
        self.x, self.y = self.waypoints[0][:2]
        self.dwell_timer = 0.0
        self.is_active = True
        self.state = "entering"

    def _build_trajectory(self):
        """Constructs realistic shopper paths through Store, Shelf, and Checkout areas."""
        # Canvas dimensions: 960 x 600
        # Store Area: (48, 90, 460, 312)
        # Shelf Area: (48, 330, 460, 540)
        # Checkout Area: (499, 90, 912, 540)
        
        entrance = (80, 50, 0.0)
        
        # Store Area browsing waypoints
        store_wp1 = (random.randint(120, 360), random.randint(140, 260), random.uniform(3.0, 6.0))
        store_wp2 = (random.randint(140, 380), random.randint(160, 270), random.uniform(2.0, 4.0))
        
        # Shelf Area inspection waypoints
        shelf_wp1 = (random.randint(120, 380), random.randint(370, 480), random.uniform(3.0, 5.0))
        shelf_wp2 = (random.randint(150, 360), random.randint(390, 490), random.uniform(2.0, 4.0))
        
        # Checkout queue waypoints (line formation)
        # Checkout registers are on the right side (~800, 200..400)
        queue_pos = 1 + (self.id % 5)
        queue_x = 750 - (queue_pos * 40)
        queue_y = 260 + (self.id % 2) * 60
        checkout_wp = (queue_x, queue_y, random.uniform(4.0, 8.0))
        
        # Register service & exit
        register_wp = (820, 260, 2.0)
        exit_wp = (920, 50, 0.0)

        if self.path_type == "rush":
            # Direct rush to checkout
            self.waypoints = [entrance, store_wp1, checkout_wp, register_wp, exit_wp]
        elif self.path_type == "shelf_focused":
            self.waypoints = [entrance, shelf_wp1, shelf_wp2, checkout_wp, register_wp, exit_wp]
        else:
            # Full store browsing: Store Area -> Shelf Area -> Checkout Area -> Exit
            self.waypoints = [entrance, store_wp1, store_wp2, shelf_wp1, checkout_wp, register_wp, exit_wp]

    def update(self, dt: float):
        """Advances customer position towards current waypoint."""
        if not self.is_active or self.current_wp_idx >= len(self.waypoints):
            self.is_active = False
            return

        tx, ty, target_dwell = self.waypoints[self.current_wp_idx]
        dx = tx - self.x
        dy = ty - self.y
        dist = math.hypot(dx, dy)

        step_dist = 65.0 * self.speed_factor * dt  # pixels per second

        if dist <= step_dist:
            self.x = tx
            self.y = ty
            # At target, accumulate dwell
            self.dwell_timer += dt
            if self.dwell_timer >= target_dwell:
                self.dwell_timer = 0.0
                self.current_wp_idx += 1
                if self.current_wp_idx >= len(self.waypoints):
                    self.is_active = False
        else:
            self.x += (dx / dist) * step_dist
            self.y += (dy / dist) * step_dist


class RetailSimulator:
    """
    Synthesizes retail store camera frames with moving customer sprites.
    Used by the existing YOLOv8 + ByteTrack pipeline to demonstrate:
    - Customer count
    - Checkout queue count
    - Customer dwell time
    - Zone occupancy
    """
    def __init__(self, width: int = 960, height: int = 600):
        self.width = width
        self.height = height
        self.customers = []
        self.customer_counter = 0
        self.sprites = []
        self.last_update_time = time.time()
        self.speed_multiplier = 1.0

        self._load_sprites()
        self._init_baseline_customers()

    def _load_sprites(self):
        """Extracts photographic person crops from Ultralytics assets."""
        bus_path = r"venv\Lib\site-packages\ultralytics\assets\bus.jpg"
        if not os.path.exists(bus_path):
            bus_path = "bus.jpg"
        
        bus_img = cv2.imread(bus_path)
        if bus_img is not None:
            # Crop 1: Tall pedestrian on left
            p1 = bus_img[398:902, 48:245]
            # Crop 2: Middle person
            p2 = bus_img[405:857, 221:344]
            # Crop 3: Person on right near bus
            p3 = bus_img[392:877, 669:809]

            target_h = 160
            for crop in [p1, p2, p3]:
                if crop.size > 0:
                    h, w = crop.shape[:2]
                    scale = target_h / float(h)
                    target_w = max(40, int(w * scale))
                    resized = cv2.resize(crop, (target_w, target_h))
                    self.sprites.append(resized)

        if not self.sprites:
            # Fallback realistic person sprite if asset is missing
            dummy = np.full((150, 60, 3), (120, 80, 50), dtype=np.uint8)
            cv2.circle(dummy, (30, 25), 18, (180, 140, 110), cv2.FILLED)
            cv2.rectangle(dummy, (10, 45), (50, 110), (180, 50, 50), cv2.FILLED)
            self.sprites.append(dummy)

    def _init_baseline_customers(self):
        """Initializes a staggered set of customers across different zones."""
        self.customers.clear()
        self.customer_counter = 0
        
        # 1 customer entering Store Area
        c1 = SimulatedCustomer(1, 0, path_type="standard", speed_factor=1.0)
        c1.x, c1.y = 180, 180
        c1.current_wp_idx = 1
        self.customers.append(c1)

        # 1 customer browsing Shelf Area
        c2 = SimulatedCustomer(2, 1 % len(self.sprites), path_type="shelf_focused", speed_factor=0.9)
        c2.x, c2.y = 240, 420
        c2.current_wp_idx = 1
        self.customers.append(c2)

        # 2 customers in Checkout Area queue
        c3 = SimulatedCustomer(3, 2 % len(self.sprites), path_type="rush", speed_factor=1.1)
        c3.x, c3.y = 670, 270
        c3.current_wp_idx = 2
        self.customers.append(c3)

        c4 = SimulatedCustomer(4, 0, path_type="rush", speed_factor=1.0)
        c4.x, c4.y = 620, 270
        c4.current_wp_idx = 2
        self.customers.append(c4)

        self.customer_counter = 4

    def spawn_customer(self, path_type: str = "standard", speed_factor: float = 1.0):
        """Spawns an additional customer entering the store."""
        self.customer_counter += 1
        sprite_idx = self.customer_counter % len(self.sprites)
        c = SimulatedCustomer(self.customer_counter, sprite_idx, path_type=path_type, speed_factor=speed_factor)
        self.customers.append(c)

    def trigger_rush_hour(self):
        """Sends multiple customers directly into the Checkout Area to demonstrate queue alert."""
        for _ in range(4):
            self.customer_counter += 1
            sprite_idx = self.customer_counter % len(self.sprites)
            c = SimulatedCustomer(self.customer_counter, sprite_idx, path_type="rush", speed_factor=1.4)
            # Spawn directly approaching checkout queue
            c.x = random.randint(530, 600)
            c.y = random.randint(220, 320)
            c.current_wp_idx = 2
            self.customers.append(c)

    def reset(self):
        """Resets the simulation to initial state."""
        self._init_baseline_customers()
        self.last_update_time = time.time()

    def get_frame(self) -> np.ndarray:
        """
        Advances customer movements and synthesizes a camera frame.
        Returns BGR image ready for ObjectDetector.detect().
        """
        curr_time = time.time()
        dt = min(0.1, max(0.01, curr_time - self.last_update_time)) * self.speed_multiplier
        self.last_update_time = curr_time

        # Update customer positions
        for c in self.customers:
            c.update(dt)

        # Remove finished customers and auto-replenish to keep scene lively
        self.customers = [c for c in self.customers if c.is_active]
        if len(self.customers) < 3:
            p_type = random.choice(["standard", "shelf_focused", "rush"])
            self.spawn_customer(path_type=p_type, speed_factor=random.uniform(0.9, 1.2))

        # Create store canvas (clean retail floor)
        frame = np.full((self.height, self.width, 3), (235, 238, 242), dtype=np.uint8)

        # Draw subtle floor tile grid
        tile_size = 60
        for x in range(0, self.width, tile_size):
            cv2.line(frame, (x, 0), (x, self.height), (220, 224, 230), 1)
        for y in range(0, self.height, tile_size):
            cv2.line(frame, (0, y), (self.width, y), (220, 224, 230), 1)

        # Draw retail floor markings
        # Entrance banner
        cv2.rectangle(frame, (30, 10), (220, 45), (210, 215, 220), cv2.FILLED)
        cv2.putText(frame, "STORE ENTRANCE", (45, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 90, 100), 1, cv2.LINE_AA)

        # Store Area floor guidance
        cv2.putText(frame, "[ AISLE 1 - GENERAL MERCHANDISE ]", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 185, 195), 1, cv2.LINE_AA)

        # Shelf Area shelf racks (gray shelf fixture visuals)
        for shelf_y in [350, 440, 520]:
            cv2.rectangle(frame, (60, shelf_y), (440, shelf_y + 14), (190, 195, 205), cv2.FILLED)
            cv2.putText(frame, "SHELF RACK", (70, shelf_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 125, 135), 1, cv2.LINE_AA)

        # Checkout counters
        cv2.rectangle(frame, (810, 160), (870, 360), (200, 205, 215), cv2.FILLED)
        cv2.rectangle(frame, (810, 160), (870, 360), (160, 165, 175), 2)
        cv2.putText(frame, "POS", (825, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (90, 95, 105), 1, cv2.LINE_AA)
        cv2.putText(frame, "LANE 1", (820, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 95, 105), 1, cv2.LINE_AA)

        # Paste simulated customer sprites
        for c in sorted(self.customers, key=lambda cust: cust.y):
            sprite = self.sprites[c.sprite_idx % len(self.sprites)]
            sh, sw = sprite.shape[:2]
            
            # Center sprite on (x, y)
            x1 = int(c.x - sw // 2)
            y1 = int(c.y - sh // 2)
            x2 = x1 + sw
            y2 = y1 + sh

            # Boundary clipping
            if x1 < 0:
                sprite = sprite[:, -x1:]
                x1 = 0
            if y1 < 0:
                sprite = sprite[-y1:, :]
                y1 = 0
            if x2 > self.width:
                sprite = sprite[:, :self.width - x1]
                x2 = self.width
            if y2 > self.height:
                sprite = sprite[:self.height - y1, :]
                y2 = self.height

            if sprite.shape[0] > 0 and sprite.shape[1] > 0:
                # Add shadow under feet
                shadow_cx = int(c.x)
                shadow_cy = min(self.height - 5, int(c.y + sh // 2 - 4))
                cv2.ellipse(frame, (shadow_cx, shadow_cy), (sw // 2, 8), 0, 0, 360, (180, 185, 190), cv2.FILLED)
                # Alpha blend or direct blit
                frame[y1:y2, x1:x2] = sprite

        return frame

