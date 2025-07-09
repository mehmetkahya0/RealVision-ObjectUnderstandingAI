"""
Create a simple icon for the RealVision AI application
This script generates a basic icon that can be used with PyInstaller
"""

try:
    from PIL import Image, ImageDraw, ImageFont
    import os
    
    def create_icon():
        # Create a 256x256 image with a blue background
        size = 256
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw a circle background
        margin = 20
        draw.ellipse([margin, margin, size-margin, size-margin], fill=(0, 120, 215, 255))
        
        # Draw the text "RV" for RealVision
        try:
            # Try to use a system font
            font = ImageFont.truetype("arial.ttf", 80)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Draw text
        text = "RV"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (size - text_width) // 2
        y = (size - text_height) // 2 - 10
        
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
        
        # Add a small camera icon effect
        camera_size = 30
        camera_x = size - camera_size - 30
        camera_y = 30
        draw.rectangle([camera_x, camera_y, camera_x + camera_size, camera_y + camera_size//2], 
                      fill=(255, 255, 255, 200))
        draw.ellipse([camera_x + 5, camera_y + 5, camera_x + camera_size - 5, camera_y + camera_size//2 - 5], 
                    fill=(0, 120, 215, 255))
        
        # Save as ICO file
        icon_path = os.path.join("media", "icon.ico")
        os.makedirs("media", exist_ok=True)
        
        # Convert to ico format
        img.save(icon_path, format='ICO', sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)])
        print(f"✓ Icon created: {icon_path}")
        
        # Also save as PNG for other uses
        png_path = os.path.join("media", "icon.png")
        img.save(png_path, format='PNG')
        print(f"✓ PNG icon created: {png_path}")
    
    if __name__ == "__main__":
        create_icon()
        
except ImportError:
    print("PIL (Pillow) not available. Skipping icon creation.")
    print("Install with: pip install Pillow")
except Exception as e:
    print(f"Error creating icon: {e}")
