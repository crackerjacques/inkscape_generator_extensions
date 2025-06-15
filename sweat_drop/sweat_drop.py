#!/usr/bin/env python3
"""
Sweat Drops Generator Extension for Inkscape
"""

import sys
import os
import random
import math

try:
    import inkex
except ImportError as e:
    sys.exit(1)

from inkex import Circle, Ellipse, Group

class SweatDropsExtension(inkex.EffectExtension):
    """Extension to automatically generate sweat drops"""
    
    def add_arguments(self, pars):
        """Parameter definitions - adjusted to match SVG color scheme"""
        # Drop settings
        pars.add_argument("--drop_count", type=int, default=5)
        pars.add_argument("--drop_size", type=float, default=20.0)
        pars.add_argument("--size_variation", type=float, default=0.3)
        
        # Shadow settings
        pars.add_argument("--shadow_angle", type=float, default=45.0)
        pars.add_argument("--shadow_distance", type=float, default=3.0)
        pars.add_argument("--shadow_opacity", type=float, default=0.515)
        
        # Highlight settings
        pars.add_argument("--highlight_angle", type=float, default=135.0)
        pars.add_argument("--highlight_size", type=float, default=0.3)
        pars.add_argument("--highlight_opacity", type=float, default=0.963)
        
        # Positioning settings
        pars.add_argument("--area_width", type=float, default=200.0)
        pars.add_argument("--area_height", type=float, default=200.0)
        
        # Blend mode settings
        pars.add_argument("--blend_mode", default="normal")
    
    def effect(self):
        """Main processing method"""
        try:
            count = self.options.drop_count
            base_size = self.options.drop_size
            size_var = self.options.size_variation
            
            current_layer = self.svg.get_current_layer()
            
            main_group = Group()
            main_group.label = "Sweat Drops"
            
            for i in range(count):
                drop_group = self.create_single_drop(i, base_size, size_var)
                main_group.add(drop_group)
            
            current_layer.add(main_group)
            
        except Exception as e:
            pass
    
    def create_single_drop(self, index, base_size, size_variation):
        """Create a single sweat drop"""
        # Determine size
        size_factor = 1.0 + random.uniform(-size_variation, size_variation)
        drop_size = base_size * size_factor
        
        # Determine position
        x = random.uniform(0, self.options.area_width)
        y = random.uniform(0, self.options.area_height)
        
        # Create drop group
        drop_group = Group()
        drop_group.label = f"Drop_{index + 1}"
        
        # Create shadow
        shadow = self.create_shadow(x, y, drop_size)
        drop_group.add(shadow)
        
        # Create main drop
        main_drop = self.create_main_drop(x, y, drop_size)
        drop_group.add(main_drop)
        
        # Create highlight
        highlight = self.create_highlight(x, y, drop_size)
        drop_group.add(highlight)
        
        return drop_group
    
    def create_shadow(self, x, y, size):
        """Create shadow using linear gradient"""
        angle_rad = math.radians(self.options.shadow_angle)
        shadow_x = x + math.cos(angle_rad) * self.options.shadow_distance
        shadow_y = y + math.sin(angle_rad) * self.options.shadow_distance
        
        shadow = Ellipse()
        shadow.set('cx', str(shadow_x))
        shadow.set('cy', str(shadow_y))
        shadow.set('rx', str(size * 0.6))
        shadow.set('ry', str(size * 0.8))
        
        shadow_gradient_id = self.create_shadow_gradient()
        
        shadow.style = {
            'fill': f'url(#{shadow_gradient_id})',
            'stroke': 'none',
            'mix-blend-mode': self.options.blend_mode
        }
        
        return shadow
    
    def create_main_drop(self, x, y, size):
        """Create main drop body (skin tone compatible)"""
        drop = Ellipse()
        drop.set('cx', str(x))
        drop.set('cy', str(y))
        drop.set('rx', str(size * 0.6))
        drop.set('ry', str(size * 0.8))
        
        gradient_id = self.create_drop_gradient()
        
        drop.style = {
            'fill': f'url(#{gradient_id})',
            'stroke': 'none',
            'mix-blend-mode': self.options.blend_mode,
            'opacity': '1'
        }
        
        return drop
    
    def create_highlight(self, x, y, size):
        """Create highlight (skin tone compatible)"""
        # Calculate highlight p
        angle_rad = math.radians(self.options.highlight_angle)
        highlight_offset = size * 0.3
        highlight_x = x + math.cos(angle_rad) * highlight_offset
        highlight_y = y + math.sin(angle_rad) * highlight_offset
        
        highlight = Ellipse()
        highlight_size = size * self.options.highlight_size
        highlight.set('cx', str(highlight_x))
        highlight.set('cy', str(highlight_y))
        highlight.set('rx', str(highlight_size * 0.4))
        highlight.set('ry', str(highlight_size * 0.6))
        
        highlight.style = {
            'fill': '#ffffff',
            'fill-opacity': str(self.options.highlight_opacity),
            'stroke': 'none',
            'mix-blend-mode': self.options.blend_mode
        }
        
        return highlight
    
    def create_shadow_gradient(self):
        """Create linear gradient for shadow"""
        # Generate gradient ID
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
        """Create gradient for drop (adjusted to match SVG colors)"""
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
        
        # Add to defs
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
        
        extension = SweatDropsExtension()
        extension.run()
        
    except Exception as e:
        pass