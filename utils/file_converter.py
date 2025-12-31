from pxr import Usd

usda_file = r"C:\Users\david\Desktop\VulkanRayTracer\scenes\cornell_box.usda"
usdc_file = r"C:\Users\david\Desktop\VulkanRayTracer\scenes\cornell_box.usdc"

# Open the USDA stage (ASCII)
stage = Usd.Stage.Open(usda_file)
if not stage:
    raise RuntimeError(f"Failed to open USDA file: {usda_file}")

# Export directly to USDC (binary) by saving as a copy
success = stage.GetRootLayer().Export(usdc_file, args={"format": "usdc"})
if not success:
    raise RuntimeError(f"Failed to export to USDC file: {usdc_file}")

print(f"Converted {usda_file} to {usdc_file} using Python API.")