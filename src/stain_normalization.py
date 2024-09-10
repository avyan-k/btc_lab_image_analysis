import loading_data as ld
import utils
import torch
from torchvision import transforms
from torchvision.utils import save_image
import torchstain
import cv2
from pathlib import Path
import os
from tqdm import tqdm
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def find_cases(image_directory):
    paths = [path for path in Path(image_directory).rglob('*.jpg')]
    cases = list(set([path.name.split('.')[0].split('_')[0] if len(path.name.split('.')[0].split('_')) <= 2 else path.name.split('.')[0].split('_')[0]+'_'+path.name.split('.')[0].split('_')[1] for path in paths]))
    # case_dict = {case:sorted([os.path.join(path.parent,path.name[:-6]+path.name[-4:]) for path in paths if case in path.name]) for case in tqdm(cases,desc="Case",position=1,leave=False)}
    case_dict = {case:sorted([os.path.join(path.parent,path.name) for path in paths if case in path.name]) for case in tqdm(cases,desc="Case",position=1,leave=False)}
    print()
    # for path in tqdm(paths):
    #     os.rename(os.path.join(path.parent,path.name),os.path.join(path.parent,path.name[:-6]+path.name[-4:]))
    return case_dict
# Create a PDF with clustered images
def create_case_pdfs_by_case(pdf_directory, case_dict, normalize = False, normalizer = None, normalization_source_path = '', source_type = ''):
    source_file = ''
    if normalize:
        source_file = os.path.split(normalization_source_path)[1][:-4] #extract case name from path
        assert normalizer is not None and os.path.isfile(normalization_source_path) and source_type != ''
    for case_id, image_paths in tqdm(case_dict.items(),desc="Case",position=1,leave=False, total=len(case_dict.keys())):
        pdf_path = os.path.join(pdf_directory,f'{case_id}_tiles.pdf')
        if normalize:
            pdf_path = os.path.join(pdf_directory,f'{case_id}_tiles_normalized_with_{source_file}.pdf')
        if not os.path.isfile(pdf_path):
            with PdfPages(pdf_path) as pdf:
                fig, ax = plt.subplots(figsize=(3, 3))
                plt.suptitle(f"Base Sample Cases", fontsize=8) 
                ax.axis('off')  # Hide axes  
                if normalize:
                    plt.suptitle(f"Reference({source_type}): {source_file}.jpg", fontsize=6) 
                    ax.imshow(Image.open(normalization_source_path))
                pdf.savefig(fig, dpi=300)  # Increase DPI to 300 for better quality
                plt.close(fig)

                images_per_page = 10 * 10
                num_pages = len(image_paths) // images_per_page + int(len(image_paths) % images_per_page > 0)
                for page in tqdm(range (num_pages),desc="Page",position=2,leave=False):
                    # Create a new figure for each page
                    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(8.5, 11))  # standard letter size
                    plt.suptitle(f"Cluster {case_id}", fontsize=16)
                    # Flatten the axes array for easy iteration
                    axes = axes.flatten()
                    
                    page_image_paths = image_paths[page * images_per_page:(page + 1) * images_per_page]
                    for i in range(len(page_image_paths)):
                        image_path = page_image_paths[i]
                        ax = axes[i]
                        if normalize:
                            img = stain_normalize(image_path,normalizer)
                        else:
                            img = Image.open(image_path)
                        # Load and display the image
                        ax.imshow(img)
                        ax.axis('off')  # Hide axes

                        base_filename = os.path.splitext(os.path.basename(image_path))[0] 

                        ax.set_title(base_filename, fontsize=3)  # Smaller font size

                    for j in range(i + 1, len(axes)): # type: ignore
                        axes[j].axis('off')
                    # Save the current figure to the PDF with high DPI
                    pdf.savefig(fig, dpi=300)  # Increase DPI to 300 for better quality
                    plt.close(fig)
            # print(f"Clusters saved to {pdf_path} as a PDF.")
    return
def create_case_pdfs(pdf_directory, case_dict, pages_per_case = -1, normalize = False, normalizer = None, normalization_source_path = '', source_type = ''):
    source_file = ''
    pdf_path = os.path.join(pdf_directory,f'sample_cases.pdf')
    if normalize:
        source_file = os.path.split(normalization_source_path)[1][:-4] #extract case name from path
        assert normalizer is not None and os.path.isfile(normalization_source_path) and source_type != ''
        pdf_path = os.path.join(pdf_directory,f'sample_cases_normalized_with_{source_type}_reference.pdf')
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(3, 3))
        plt.suptitle(f"Base Sample Cases", fontsize=8) 
        ax.axis('off')  # Hide axes  
        if normalize:
            plt.suptitle(f"Reference({source_type}): {source_file}.jpg", fontsize=6)
            ax.imshow(Image.open(normalization_source_path))
        pdf.savefig(fig, dpi=300)  # Increase DPI to 300 for better quality
        plt.close(fig)
        for case_id, image_paths in tqdm(case_dict.items(),desc="Case",position=1,leave=False, total=len(case_dict.keys())):

                images_per_page = 10 * 10

                num_pages = len(image_paths) // images_per_page + int(len(image_paths) % images_per_page > 0)
                if pages_per_case > 0:
                    num_pages = pages_per_case
                for page in tqdm(range (num_pages),desc="Page",position=2,leave=False):
                    # Create a new figure for each page
                    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(8.5, 11))  # standard letter size
                    plt.suptitle(f"Case/Slide {case_id}", fontsize=16)
                    # Flatten the axes array for easy iteration
                    axes = axes.flatten()
                    # Get the images for the current page
                    page_image_paths = image_paths[page * images_per_page:(page + 1) * images_per_page]
                    for i in range(len(page_image_paths)):
                        image_path = page_image_paths[i]
                        ax = axes[i]
                        if normalize:
                            img = stain_normalize(image_path,normalizer)
                        else:
                            img = Image.open(image_path)
                        # Load and display the image
                        ax.imshow(img)
                        ax.axis('off')  # Hide axes
                        # Get the base filename and remove the .jpg extension
                        base_filename = os.path.splitext(os.path.basename(image_path))[0] 
                        # Display the filename without the extension in a smaller font
                        ax.set_title(base_filename, fontsize=3)  # Smaller font size

                    for j in range(i + 1, len(axes)): # type: ignore
                        axes[j].axis('off')
                    # Save the current figure to the PDF with high DPI
                    pdf.savefig(fig, dpi=300)  # Increase DPI to 300 for better quality
                    plt.close(fig)
            # print(f"Clusters saved to {pdf_path} as a PDF.")
    return
def stain_normalize(image_path, normalizer, save = False, save_path = ''):
    
    T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
    ])
    to_transform = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    t_to_transform = T(to_transform)
    norm,H,E = normalizer.normalize(I=t_to_transform, stains = False)
    if save:
        assert save_path != ''
        # print(image_file,save_directory)
        save_image(tensor=torch.permute(norm/255.0,(2,0,1)),fp=save_path)
        return
    shrinked_image = cv2.resize(norm,(256,256),interpolation=cv2.INTER_AREA)
    pil_image = transforms.functional.to_pil_image(torch.permute(shrinked_image/255.0,(2,0,1)),mode = 'RGB')
    return pil_image

if __name__ == "__main__":
    tumor_type = "SCCOHT_1"
    page_sampled_per_case = 2

    image_directory = f"./images/{tumor_type}/images"
    
    result_directory = f"./results/Cases/{tumor_type}"
    Path(result_directory).mkdir(parents=True, exist_ok=True)
    normalize_directory = f"./images/{tumor_type}/normalized_images"
    annotations = ['normal', 'undiff', 'well_diff'] if 'DDC_UC' in tumor_type else ['normal', 'tumor']
    for annotations in annotations:
        Path(os.path.join(normalize_directory,annotations)).mkdir(parents=True, exist_ok=True)


    image_paths = [str(path) for path in Path(image_directory).rglob('*.jpg')]
    parent_list = [os.path.normpath(path).split(os.sep) for path in Path(image_directory).rglob('*.jpg')]
    for parents in parent_list:
        parents[2] = "normalized_images"
    norm_paths = [os.path.join(*parents) for parents in parent_list]
    
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256,256),antialias=True),
    transforms.Lambda(lambda x: x*255)
    ])
    reference_slide_path = os.path.join(image_directory,'normal', '15D16367_D3_247sn.jpg')
    normalization_source = cv2.cvtColor(cv2.imread(reference_slide_path), cv2.COLOR_BGR2RGB)
    normalizer.fit(T(normalization_source))
#     image_paths = [r'images\SCCOHT_1\images\tumor\AS17036383_A49_58057st.jpg',
# r'images\SCCOHT_1\images\tumor\PCS281092299G_92675st.jpg']
#     norm_paths = [r'images\SCCOHT_1\normalized_images\tumor\AS17036383_A49_58057st.jpg',
# r'images\SCCOHT_1\normalized_images\tumor\PCS281092299G_92675st.jpg']
    # with open(file=f"./results/{tumor_type}_unnormalizable_images.txt",mode='w') as f:
    #     f.write(f"Checked on {utils.get_time()}\n")
    #     for i,path in enumerate(tqdm(image_paths)):
    #         if not os.path.isfile(norm_paths[i]):
    #             try:
    #                 stain_normalize(path,normalizer,save=True, save_path = norm_paths[i])
    #             except:
    #                 f.write(str(path)+'\n')



    normalization_dictionnary = { 
        'SCCOHT_1' : ({
            'bright' : '15D16367_D3_247sn.jpg',
            'dark' : 'GS986_1B_78169st.jpg',
            'indigo' : '22B33007_A3_34208st.jpg'
        },['Q121348773B','AS21032651_D1','0079682','H107043','22B33007_A3','GS986_1B','524996_A4']),
        'vMRT' : ({
            'dark' : 'AC23029182_A1_1_52vn.jpg'
        },[]),
        'DDC_UC_1' : ({

        },[])
    }

    case_dict = find_cases(image_directory)
    create_case_pdfs_by_case(result_directory,case_dict,normalize = False)

    references, samples_cases = normalization_dictionnary[tumor_type]
    case_dict = find_cases(image_directory)

    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256,256),antialias=True),
    transforms.Lambda(lambda x: x*255)
    ])
    if not os.path.isfile(os.path.join(result_directory,f'sample_cases.pdf')):
        create_case_pdfs(result_directory,case_dict,pages_per_case=page_sampled_per_case,normalize = False)     
    for reference_type, reference_slide in references.items():
        annotation_prefix = reference_slide.split('.')[0][-1]
        if tumor_type in "DDC_UC":
            assert annotation_prefix in 'nuw' # annotation is either normal, undifferentiated or well-differentiated
            match annotation_prefix:
                case 'n': annotation = 'normal'
                case 'u': annotation = 'undiff'
                case 'w': annotation = 'well_diff'
        else:
            assert annotation_prefix in 'nt' # annotation is either tumor or normal
            annotation = 'normal' if annotation_prefix == 'n' else 'tumor'
        reference_slide_path = os.path.join(image_directory,annotation, reference_slide) # type: ignore
        # print(reference_slide_path)
        normalization_source = cv2.cvtColor(cv2.imread(reference_slide_path), cv2.COLOR_BGR2RGB)
        normalizer.fit(T(normalization_source))

        if not os.path.isfile(os.path.join(result_directory,f'sample_cases_normalized_with_{reference_type}_reference.pdf')):
            create_case_pdfs(result_directory,case_dict,pages_per_case=page_sampled_per_case,normalize = True, normalizer=normalizer,normalization_source_path=reference_slide_path,source_type=reference_type)
    

    

    
    
    # paths = [str(path) for path in Path(image_directory).rglob('*.jpg')][:50]
    
    
    
    
    # img = stain_normalize(os.path.join(image_directory,'tumor','S173033_A17_SC11_100959st.jpg'), normalizer)
    # img.show()
    # create_case_pdfs(result_directory,case_df,normalize = False)

    
    # 
    

    # norms = stain_normalize(f'images/SCCOHT_1/images/normal/15D16367_D3_247sn.jpg', paths, normalizer)
