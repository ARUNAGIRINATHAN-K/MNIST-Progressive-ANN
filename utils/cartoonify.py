import torch
from torchvision import transforms
from PIL import Image
from model.cartoon_gan import CartoonGenerator  # your GAN class

def cartoonify_image(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CartoonGenerator()
    model.load_state_dict(torch.load("model/cartoon_gan_model.pth", map_location=device))
    model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    img = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)[0].cpu()

    output_image = transforms.ToPILImage()(output_tensor.clamp(0, 1))
    return output_image
