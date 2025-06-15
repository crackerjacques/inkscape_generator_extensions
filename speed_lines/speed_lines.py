#!/usr/bin/env python3
"""
Speed Lines Generator Extension for Inkscape
"""

import sys
import os
import random
import math

try:
    import inkex
except ImportError as e:
    sys.exit(1)

from inkex import Group, PathElement, Circle, Ellipse, Rectangle

class SpeedLinesExtension(inkex.EffectExtension):
    """Extension to automatically generate manga-style speed lines"""
    
    def add_arguments(self, pars):
        """Parameter definitions"""
        # Speed line settings
        pars.add_argument("--line_count", type=int, default=30)
        pars.add_argument("--line_thickness_start", type=float, default=3.0)
        pars.add_argument("--line_thickness_end", type=float, default=0.5)
        pars.add_argument("--line_length", type=float, default=200.0)
        pars.add_argument("--line_variation", type=float, default=0.3)
        
        # Center space settings
        pars.add_argument("--center_shape", default="circle")
        pars.add_argument("--center_size_x", type=float, default=80.0)
        pars.add_argument("--center_size_y", type=float, default=80.0)
        
        # Center offset settings
        pars.add_argument("--center_offset_x", type=float, default=0.0)
        pars.add_argument("--center_offset_y", type=float, default=0.0)
        
        # Direction settings
        pars.add_argument("--direction", default="outward")
        
        # Angle settings
        pars.add_argument("--angle_start", type=float, default=0.0)
        pars.add_argument("--angle_range", type=float, default=360.0)
        pars.add_argument("--angle_randomness", type=float, default=5.0)
        
        # Style settings
        pars.add_argument("--line_color", default="#000000ff")
        pars.add_argument("--line_opacity", type=float, default=1.0)
        pars.add_argument("--blend_mode", default="normal")
        
        # Canvas settings
        pars.add_argument("--canvas_width", type=float, default=400.0)
        pars.add_argument("--canvas_height", type=float, default=400.0)
    
    def effect(self):
        """Main processing method"""
        try:
            current_layer = self.svg.get_current_layer()
            
            main_group = Group()
            main_group.label = "Speed Lines"
            
            # Calculate center point
            center_x = self.options.canvas_width / 2 + self.options.center_offset_x
            center_y = self.options.canvas_height / 2 + self.options.center_offset_y
            
            # Create center space shape (optional visualization)
            center_shape = self.create_center_shape(center_x, center_y)
            if center_shape is not None:
                main_group.add(center_shape)
            
            # Create speed lines
            for i in range(self.options.line_count):
                line = self.create_speed_line(i, center_x, center_y)
                if line is not None:
                    main_group.add(line)
            
            current_layer.add(main_group)
            
        except Exception as e:
            pass
    
    def create_center_shape(self, center_x, center_y):
        """Create center space shape (for reference, can be made invisible)"""
        shape = None
        
        if self.options.center_shape == "circle":
            shape = Circle()
            radius = min(self.options.center_size_x, self.options.center_size_y) / 2
            shape.set('cx', str(center_x))
            shape.set('cy', str(center_y))
            shape.set('r', str(radius))
            
        elif self.options.center_shape == "ellipse":
            shape = Ellipse()
            shape.set('cx', str(center_x))
            shape.set('cy', str(center_y))
            shape.set('rx', str(self.options.center_size_x / 2))
            shape.set('ry', str(self.options.center_size_y / 2))
            
        elif self.options.center_shape in ["square", "rectangle"]:
            shape = Rectangle()
            shape.set('x', str(center_x - self.options.center_size_x / 2))
            shape.set('y', str(center_y - self.options.center_size_y / 2))
            shape.set('width', str(self.options.center_size_x))
            shape.set('height', str(self.options.center_size_y))
            
        elif self.options.center_shape == "star":
            shape = self.create_star_shape(center_x, center_y)
        
        if shape is not None:
            shape.style = {
                'fill': 'none',
                'stroke': '#cccccc',
                'stroke-width': '1',
                'stroke-dasharray': '5,5',
                'opacity': '0.3'
            }
        
        return shape
    
    def create_star_shape(self, center_x, center_y):
        """Create star-shaped center space"""
        star_points = 5
        outer_radius = max(self.options.center_size_x, self.options.center_size_y) / 2
        inner_radius = outer_radius * 0.4
        
        path_data = []
        
        for i in range(star_points * 2):
            angle = (i * math.pi) / star_points
            if i % 2 == 0:  # Outer point
                x = center_x + outer_radius * math.cos(angle - math.pi / 2)
                y = center_y + outer_radius * math.sin(angle - math.pi / 2)
            else:  # Inner point
                x = center_x + inner_radius * math.cos(angle - math.pi / 2)
                y = center_y + inner_radius * math.sin(angle - math.pi / 2)
            
            if i == 0:
                path_data.append(f"M {x},{y}")
            else:
                path_data.append(f"L {x},{y}")
        
        path_data.append("Z")
        
        star = PathElement()
        star.set('d', " ".join(path_data))
        
        return star
    
    def create_speed_line(self, index, center_x, center_y):
        """Create a single speed line"""
        # Calculate angle
        angle_step = self.options.angle_range / self.options.line_count
        base_angle = self.options.angle_start + (index * angle_step)
        
        # Add randomness to angle
        angle_offset = random.uniform(-self.options.angle_randomness, self.options.angle_randomness)
        angle = math.radians(base_angle + angle_offset)
        
        # Calculate line length with variation
        length_variation = random.uniform(-self.options.line_variation, self.options.line_variation)
        line_length = self.options.line_length * (1.0 + length_variation)
        
        # Calculate center space boundary
        center_boundary = self.get_center_boundary(center_x, center_y, angle)
        
        # Calculate start and end points based on direction
        if self.options.direction == "outward":
            start_x = center_x + center_boundary * math.cos(angle)
            start_y = center_y + center_boundary * math.sin(angle)
            end_x = start_x + line_length * math.cos(angle)
            end_y = start_y + line_length * math.sin(angle)
            thick_start = True
            
        elif self.options.direction == "inward":
            end_x = center_x + center_boundary * math.cos(angle)
            end_y = center_y + center_boundary * math.sin(angle)
            start_x = end_x + line_length * math.cos(angle)
            start_y = end_y + line_length * math.sin(angle)
            thick_start = True
            
        else:  # both directions
            half_length = line_length / 2
            start_x = center_x + (center_boundary + half_length) * math.cos(angle)
            start_y = center_y + (center_boundary + half_length) * math.sin(angle)
            end_x = center_x - (center_boundary + half_length) * math.cos(angle)
            end_y = center_y - (center_boundary + half_length) * math.sin(angle)
            thick_start = False  # Both ends thin, thick in middle
        
        # Create tapered line using path
        line_path = self.create_tapered_line(start_x, start_y, end_x, end_y, thick_start)
        
        return line_path
    
    def get_center_boundary(self, center_x, center_y, angle):
        """Calculate distance from center to boundary of center space"""
        if self.options.center_shape == "circle":
            return min(self.options.center_size_x, self.options.center_size_y) / 2
            
        elif self.options.center_shape == "ellipse":
            # Ellipse boundary calculation
            a = self.options.center_size_x / 2  # semi-major axis
            b = self.options.center_size_y / 2  # semi-minor axis
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            
            # Distance to ellipse boundary
            return (a * b) / math.sqrt((b * cos_angle) ** 2 + (a * sin_angle) ** 2)
            
        elif self.options.center_shape in ["square", "rectangle"]:
            # Rectangle boundary calculation
            half_width = self.options.center_size_x / 2
            half_height = self.options.center_size_y / 2
            
            cos_angle = abs(math.cos(angle))
            sin_angle = abs(math.sin(angle))
            
            if cos_angle == 0:
                return half_height
            elif sin_angle == 0:
                return half_width
            else:
                # Distance to rectangle boundary
                return min(half_width / cos_angle, half_height / sin_angle)
                
        elif self.options.center_shape == "star":
            # Simplified star boundary (use average radius)
            return max(self.options.center_size_x, self.options.center_size_y) / 4
        
        return 40  # Default fallback
    
    def create_tapered_line(self, start_x, start_y, end_x, end_y, thick_start=True):
        """Create a tapered line using SVG path"""
        if self.options.direction == "both":
            # Special case for both directions - thick in middle
            return self.create_double_tapered_line(start_x, start_y, end_x, end_y)
        
        # Calculate line direction
        line_length = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
        if line_length == 0:
            return None
            
        # Unit vector along the line
        ux = (end_x - start_x) / line_length
        uy = (end_y - start_y) / line_length
        
        # Perpendicular vector
        px = -uy
        py = ux
        
        # Thickness values
        if thick_start:
            start_thickness = self.options.line_thickness_start / 2
            end_thickness = self.options.line_thickness_end / 2
        else:
            start_thickness = self.options.line_thickness_end / 2
            end_thickness = self.options.line_thickness_start / 2
        
        # Calculate path points
        start_top_x = start_x + px * start_thickness
        start_top_y = start_y + py * start_thickness
        start_bottom_x = start_x - px * start_thickness
        start_bottom_y = start_y - py * start_thickness
        
        end_top_x = end_x + px * end_thickness
        end_top_y = end_y + py * end_thickness
        end_bottom_x = end_x - px * end_thickness
        end_bottom_y = end_y - py * end_thickness
        
        # Create path data
        path_data = f"""
        M {start_top_x},{start_top_y}
        L {end_top_x},{end_top_y}
        L {end_bottom_x},{end_bottom_y}
        L {start_bottom_x},{start_bottom_y}
        Z
        """
        
        path_element = PathElement()
        path_element.set('d', " ".join(path_data.split()))
        
        # Convert color format
        color = self.options.line_color
        if len(color) == 9 and color.startswith('#'):  # #RRGGBBaa format
            color = color[:7]  # Remove alpha component
        
        path_element.style = {
            'fill': color,
            'stroke': 'none',
            'fill-opacity': str(self.options.line_opacity),
            'mix-blend-mode': self.options.blend_mode
        }
        
        return path_element
    
    def create_double_tapered_line(self, start_x, start_y, end_x, end_y):
        """Create line that's thick in the middle and thin at both ends"""
        # Calculate midpoint
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        # Calculate line direction
        line_length = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
        if line_length == 0:
            return None
            
        # Unit vector along the line
        ux = (end_x - start_x) / line_length
        uy = (end_y - start_y) / line_length
        
        # Perpendicular vector
        px = -uy
        py = ux
        
        # Thickness values
        end_thickness = self.options.line_thickness_end / 2
        mid_thickness = self.options.line_thickness_start / 2
        
        # Calculate path points for diamond-like shape
        start_x_adj = start_x
        start_y_adj = start_y
        end_x_adj = end_x
        end_y_adj = end_y
        
        mid_top_x = mid_x + px * mid_thickness
        mid_top_y = mid_y + py * mid_thickness
        mid_bottom_x = mid_x - px * mid_thickness
        mid_bottom_y = mid_y - py * mid_thickness
        
        # Create path data for diamond shape
        path_data = f"""
        M {start_x_adj},{start_y_adj}
        L {mid_top_x},{mid_top_y}
        L {end_x_adj},{end_y_adj}
        L {mid_bottom_x},{mid_bottom_y}
        Z
        """
        
        path_element = PathElement()
        path_element.set('d', " ".join(path_data.split()))
        
        # Convert color format
        color = self.options.line_color
        if len(color) == 9 and color.startswith('#'):  # #RRGGBBaa format
            color = color[:7]  # Remove alpha component
        
        path_element.style = {
            'fill': color,
            'stroke': 'none',
            'fill-opacity': str(self.options.line_opacity),
            'mix-blend-mode': self.options.blend_mode
        }
        
        return path_element

if __name__ == '__main__':
    try:
        if len(sys.argv) == 1:
            test_svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="0 0 400 400">
    <g id="layer1"></g>
</svg>'''
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
                f.write(test_svg)
                sys.argv.append(f.name)
        
        extension = SpeedLinesExtension()
        extension.run()
        
    except Exception as e:
        pass