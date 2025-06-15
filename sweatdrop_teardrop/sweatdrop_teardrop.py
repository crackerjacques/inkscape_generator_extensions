#!/usr/bin/env python3
"""
Teardrop Sweat Drops Generator Extension for Inkscape
"""

import sys
import os
import random
import math

try:
    import inkex
except ImportError as e:
    sys.exit(1)

from inkex import Group, PathElement

class TeardropSweatDropsExtension(inkex.EffectExtension):
    """Extension to automatically generate teardrop-shaped sweat drops"""
    
    def add_arguments(self, pars):
        """Parameter definitions"""
        # Drop settings
        pars.add_argument("--drop_count", type=int, default=5)
        pars.add_argument("--drop_size", type=float, default=20.0)
        pars.add_argument("--size_variation", type=float, default=0.3)
        pars.add_argument("--teardrop_ratio", type=float, default=1.5)
        
        # Shadow settings
        pars.add_argument("--shadow_angle", type=float, default=45.0)
        pars.add_argument("--shadow_distance", type=float, default=3.0)
        pars.add_argument("--shadow_opacity", type=float, default=0.2)
        
        # Highlight settings
        pars.add_argument("--highlight_angle", type=float, default=135.0)
        pars.add_argument("--highlight_size", type=float, default=0.3)
        pars.add_argument("--highlight_opacity", type=float, default=1.0)
        
        # Positioning settings
        pars.add_argument("--area_width", type=float, default=200.0)
        pars.add_argument("--area_height", type=float, default=200.0)
        
        # Shape settings
        pars.add_argument("--rotation_variation", type=float, default=30.0)
        
        pars.add_argument("--blend_mode", default="hard-light")
    
    def effect(self):
        """Main processing method"""
        try:
            count = self.options.drop_count
            base_size = self.options.drop_size
            size_var = self.options.size_variation
            
            current_layer = self.svg.get_current_layer()
            
            main_group = Group()
            main_group.label = "Teardrop Sweat Drops"
            
            for i in range(count):
                drop_group = self.create_single_teardrop(i, base_size, size_var)
                main_group.add(drop_group)
            
            current_layer.add(main_group)
            
        except Exception as e:
            pass
    
    def create_single_teardrop(self, index, base_size, size_variation):
        """Create a single teardrop"""
        # Determine size
        size_factor = 1.0 + random.uniform(-size_variation, size_variation)
        drop_size = base_size * size_factor
        
        # Determine position
        x = random.uniform(0, self.options.area_width)
        y = random.uniform(0, self.options.area_height)
        
        # Determine rotation angle
        rotation = random.uniform(-self.options.rotation_variation, self.options.rotation_variation)
        
        # Create teardrop group
        drop_group = Group()
        drop_group.label = f"Teardrop_{index + 1}"
        
        # Create shadow
        shadow = self.create_teardrop_shadow(x, y, drop_size, rotation)
        drop_group.add(shadow)
        
        # Create main teardrop
        main_drop = self.create_teardrop_main(x, y, drop_size, rotation)
        drop_group.add(main_drop)
        
        # Create highlight
        highlight = self.create_teardrop_highlight(x, y, drop_size, rotation)
        drop_group.add(highlight)
        
        return drop_group
    
    def create_teardrop_path(self, x, y, size, rotation=0):
        """Create teardrop-shaped path"""
        # Basic shape parameters
        width = size * 0.6
        height = size * self.options.teardrop_ratio
        
        # Calculate control points
        bottom_y = y + height * 0.3
        top_y = y - height * 0.7
        
        # Bezier curve control points
        cx1 = x - width * 0.5
        cx2 = x + width * 0.5
        cy_mid = y
        
        # Build path data (SVG d attribute)
        path_data = f"""
        M {x},{bottom_y}
        C {cx1},{bottom_y} {cx1},{cy_mid} {cx1},{cy_mid - height * 0.2}
        C {cx1},{top_y + height * 0.3} {x - width * 0.2},{top_y + height * 0.1} {x},{top_y}
        C {x + width * 0.2},{top_y + height * 0.1} {cx2},{top_y + height * 0.3} {cx2},{cy_mid - height * 0.2}
        C {cx2},{cy_mid} {cx2},{bottom_y} {x},{bottom_y}
        Z
        """
        
        # Clean up path data
        path_data = " ".join(path_data.split())
        
        return path_data
    
    def create_teardrop_shadow(self, x, y, size, rotation):
        """Create teardrop shadow"""
        angle_rad = math.radians(self.options.shadow_angle)
        shadow_x = x + math.cos(angle_rad) * self.options.shadow_distance
        shadow_y = y + math.sin(angle_rad) * self.options.shadow_distance
        
        path_data = self.create_teardrop_path(shadow_x, shadow_y, size, rotation)
        
        shadow_path = PathElement()
        shadow_path.set('d', path_data)
        
        shadow_gradient_id = self.create_shadow_gradient()
        
        shadow_path.style = {
            'fill': f'url(#{shadow_gradient_id})',
            'stroke': 'none',
            'mix-blend-mode': self.options.blend_mode
        }
        
        if rotation != 0:
            shadow_path.transform = inkex.Transform(rotate=(rotation, shadow_x, shadow_y))
        
        return shadow_path
    
    def create_teardrop_main(self, x, y, size, rotation):
        """Create main teardrop body"""
        path_data = self.create_teardrop_path(x, y, size, rotation)
        
        main_path = PathElement()
        main_path.set('d', path_data)
        
        gradient_id = self.create_drop_gradient()
        
        main_path.style = {
            'fill': f'url(#{gradient_id})',
            'stroke': 'none',
            'mix-blend-mode': self.options.blend_mode,
            'opacity': '1'
        }
        
        if rotation != 0:
            main_path.transform = inkex.Transform(rotate=(rotation, x, y))
        
        return main_path
    
    def create_teardrop_highlight(self, x, y, size, rotation):
        """Create teardrop highlight"""
        # Calculate highlight position (adjusted for teardrop shape)
        angle_rad = math.radians(self.options.highlight_angle)
        highlight_offset = size * 0.2
        highlight_x = x + math.cos(angle_rad) * highlight_offset
        highlight_y = y + math.sin(angle_rad) * highlight_offset
        
        highlight_size = size * self.options.highlight_size
        path_data = self.create_teardrop_path(highlight_x, highlight_y, highlight_size, rotation)
        
        highlight_path = PathElement()
        highlight_path.set('d', path_data)
        
        highlight_path.style = {
            'fill': '#ffffff',
            'fill-opacity': str(self.options.highlight_opacity),
            'stroke': 'none',
            'mix-blend-mode': self.options.blend_mode
        }
        
        if rotation != 0:
            highlight_path.transform = inkex.Transform(rotate=(rotation, highlight_x, highlight_y))
        
        return highlight_path
    
    def create_shadow_gradient(self):
        """Create linear gradient for shadow"""
        gradient_id = f"shadowGradient_{random.randint(1000, 9999)}"
        
        gradient = inkex.LinearGradient()
        gradient.set('id', gradient_id)
        
        stop1 = inkex.Stop()
        stop1.set('offset', '0%')
        stop1.style = {'stop-color': '#000000', 'stop-opacity': str(self.options.shadow_opacity)}
        
        stop2 = inkex.Stop()
        stop2.set('offset', '100%')
        stop2.style = {'stop-color': '#000000', 'stop-opacity': '0.0'}
        
        gradient.add(stop1)
        gradient.add(stop2)
        
        defs = self.svg.defs
        defs.add(gradient)
        
        return gradient_id
    
    def create_drop_gradient(self):
        """Create gradient for teardrop"""
        # Generate gradient ID
        gradient_id = f"dropGradient_{random.randint(1000, 9999)}"
        
        gradient = inkex.RadialGradient()
        gradient.set('id', gradient_id)
        gradient.set('cx', '0.3')
        gradient.set('cy', '0.3')
        gradient.set('r', '0.7')
        
        stop1 = inkex.Stop()
        stop1.set('offset', '0%')
        stop1.style = {'stop-color': '#ffffff', 'stop-opacity': '0.0'}
        
        stop2 = inkex.Stop()
        stop2.set('offset', '37%')
        stop2.style = {'stop-color': '#121212', 'stop-opacity': '0.014'}
        
        stop3 = inkex.Stop()
        stop3.set('offset', '100%')
        stop3.style = {'stop-color': '#ffffff', 'stop-opacity': '0.286'}
        
        gradient.add(stop1)
        gradient.add(stop2)
        gradient.add(stop3)
        
        defs = self.svg.defs
        defs.add(gradient)
        
        return gradient_id

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
        
        extension = TeardropSweatDropsExtension()
        extension.run()
        
    except Exception as e:
        pass