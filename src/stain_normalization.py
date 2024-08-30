import loading_data as ld
import torch
from torchvision import transforms
import torchstain
import cv2
from pathlib import Path
import os
from tqdm import tqdm
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.colors import yellow, green, blue, black
from reportlab.lib.utils import ImageReader
def find_all_cases(image_directory):
    paths = [path for path in Path(image_directory).rglob('*.jpg')]
    cases = list(set([path.name.split('.')[0].split('_')[0] if len(path.name.split('.')[0].split('_')) <= 2 else path.name.split('.')[0].split('_')[0]+'_'+path.name.split('.')[0].split('_')[1] for path in paths]))
    # case_dict = {case:sorted([os.path.join(path.parent,path.name[:-6]+path.name[-4:]) for path in paths if case in path.name]) for case in tqdm(cases,desc="Case",position=1,leave=False)}
    case_dict = {case:sorted([os.path.join(path.parent,path.name) for path in paths if case in path.name]) for case in tqdm(cases,desc="Case",position=1,leave=False)}
    print()
    # for path in tqdm(paths):
    #     os.rename(os.path.join(path.parent,path.name),os.path.join(path.parent,path.name[:-6]+path.name[-4:]))
    return case_dict
# Create a PDF with clustered images
def create_case_pdfs(pdf_path, case_dictionnary,normalize = False, normalizer = None):
    if normalize:
        assert normalizer is not None
    width, height = letter
    x, y = 30, height - 80
    # Define tile size and margins
    tile_size = (50, 50)  # Reduced tile size
    margin = 5
    styles = getSampleStyleSheet()
    small_font = styles['BodyText']
    small_font.fontSize = 4  # Smaller font size for filenames
    for case_id, images in tqdm(case_dictionnary.items(),desc="Case",position=1,leave=False):
        c = canvas.Canvas(os.path.join(pdf_path,f"{case_id}_tiles.pdf"), pagesize=letter,pageCompression=1)
        c.setFont(small_font.fontName, small_font.fontSize)
        c.setFillColor(black)
        # c.drawString(30, height - 30, f"Cluster {case_id}")

        for image_path in tqdm(images,desc="Tile",position=2,leave=False):
            if normalize:
                img = stain_normalize(image_path,normalizer)
            else:
                img = Image.open(image_path)
                
            img.resize(tile_size,Image.LANCZOS)
            c.drawImage(image=ImageReader(img), x=x, y=y, width=tile_size[0], height=tile_size[1])
            c.setFillColor(black)
            c.setFont(small_font.fontName, small_font.fontSize)
            c.drawString(x+1, y + tile_size[1] + 1, os.path.splitext(os.path.basename(image_path))[0])
            if not normalize: # close image taken from disk
                img.close()
            x += tile_size[0] + margin
            if x + tile_size[0] > width - 30:
                x = 30
                y -= tile_size[1] + margin
                if y < 100:
                    c.showPage()
                    c.setFont(small_font.fontName, small_font.fontSize)
                    c.setFillColor(black)
                    x, y = 30, height - 80

        c.save()
# save = False, save_path = ""
def stain_normalize(image_path, normalizer):
    T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256,256)),
    transforms.Lambda(lambda x: x*255)
    ])
    to_transform = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    t_to_transform = T(to_transform)
    norm,H,E = normalizer.normalize(I=t_to_transform, stains = False)
    return transforms.functional.to_pil_image(torch.permute(norm,(2,0,1)),mode = 'RGB')

if __name__ == "__main__":
    tumor_type = "SCCOHT_1"
    image_directory = f"./images/{tumor_type}/images"

    reference_slide_path = os.path.join(image_directory,'normal','15D16367_D3_247sn.jpg')
    dictionary = find_all_cases(image_directory)
    result_directory = f"./results/Cases/{tumor_type}"
    Path(result_directory).mkdir(parents=True, exist_ok=True)
    paths = [str(path) for path in Path(image_directory).rglob('*.jpg')][:50]
    normalization_source = cv2.cvtColor(cv2.imread(reference_slide_path), cv2.COLOR_BGR2RGB)
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256,256)),
    transforms.Lambda(lambda x: x*255)
    ])
    normalizer.fit(T(normalization_source))
    create_case_pdfs(result_directory,dictionary,normalize = True, normalizer=normalizer)

    # norms = stain_normalize(f'images/SCCOHT_1/images/normal/15D16367_D3_247sn.jpg', paths, normalizer)
